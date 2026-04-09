/*
 * MVKCmdRayTracing.mm
 *
 * Copyright (c) 2015-2025 The Brenwill Workshop Ltd. (http://www.brenwill.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MVKCmdRayTracing.h"
#include "MVKCommandBuffer.h"
#include "MVKAccelerationStructure.h"
#include "MVKPipeline.h"
#include "MVKCommandPool.h"
#include "MVKDescriptorSet.h"
#include <vector>


static std::vector<uint32_t> mvkDecodeShaderBindingTable(MVKDevice* device, const VkStridedDeviceAddressRegionKHR& region) {
	std::vector<uint32_t> groupIndices;
	if (!region.deviceAddress || !region.size || !region.stride) { return groupIndices; }

	uint32_t recordCount = static_cast<uint32_t>(region.size / region.stride);
	if (!recordCount) { return groupIndices; }

	VkDeviceSize baseOffset = 0;
	id<MTLBuffer> mtlBuffer = device->getMTLBufferForDeviceAddress(region.deviceAddress, &baseOffset);
	const uint8_t* bufferContents = mtlBuffer ? (const uint8_t*)mtlBuffer.contents : nullptr;
	if (!bufferContents || mtlBuffer.length < baseOffset) { return groupIndices; }

	groupIndices.resize(recordCount, 0);
	const uint8_t* sbtData = bufferContents + baseOffset;
	size_t availableBytes = mtlBuffer.length - baseOffset;
	for (uint32_t i = 0; i < recordCount; i++) {
		size_t recordOffset = i * region.stride;
		if (recordOffset + sizeof(uint32_t) > availableBytes) { break; }
		memcpy(&groupIndices[i], sbtData + recordOffset, sizeof(uint32_t));
	}
	return groupIndices;
}

static void mvkBindShaderBindingTable(MVKCommandEncoder* cmdEncoder,
									  id<MTLComputeCommandEncoder> mtlEncoder,
									  uint32_t bufferIndex,
									  const std::vector<uint32_t>& groupIndices) {
	uint32_t zero = 0;
	if (groupIndices.empty()) {
		cmdEncoder->setComputeBytes(mtlEncoder, &zero, sizeof(zero), bufferIndex);
	} else {
		cmdEncoder->setComputeBytes(mtlEncoder, groupIndices.data(), groupIndices.size() * sizeof(uint32_t), bufferIndex);
	}
}


#pragma mark -
#pragma mark MVKCmdTraceRays

VkResult MVKCmdTraceRays::setContent(MVKCommandBuffer* cmdBuff,
                                     const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
                                     const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
                                     const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
                                     const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t depth) {
	_raygenSBT = *pRaygenShaderBindingTable;
	_missSBT = *pMissShaderBindingTable;
	_hitSBT = *pHitShaderBindingTable;
	_callableSBT = *pCallableShaderBindingTable;
	_width = width;
	_height = height;
	_depth = depth;
	return VK_SUCCESS;
}

void MVKCmdTraceRays::encode(MVKCommandEncoder* cmdEncoder) {
	auto* rtPipeline = cmdEncoder->getRayTracingPipeline();
	if (!rtPipeline || !rtPipeline->getMTLComputePipelineState()) { return; }

	// Finalize descriptor set state, then set the raygen pipeline state.
	cmdEncoder->finalizeDispatchState();
	id<MTLComputeCommandEncoder> mtlEncoder = cmdEncoder->getMTLComputeEncoder(kMVKCommandUseDispatch);

	[mtlEncoder setComputePipelineState: rtPipeline->getMTLComputePipelineState()];

	std::vector<uint32_t> hitGroupIndices;
	if (_hitSBT.deviceAddress && _hitSBT.size && _hitSBT.stride &&
		(rtPipeline->needsHitShaderBindingTable() || rtPipeline->usesIntersectionFunctionTable())) {
		hitGroupIndices = mvkDecodeShaderBindingTable(cmdEncoder->getDevice(), _hitSBT);
	}

	id<MTLBuffer> instanceSBTOffsetBuffer = nil;
	auto& vk = cmdEncoder->getState().vkCompute();
	if (vk._layout) {
		for (uint32_t s = 0; s < vk._layout->getDescriptorSetCount(); s++) {
			MVKDescriptorSet* set = vk._descriptorSets[s];
			const MVKDescriptorSetLayout* setLayout = vk._layout->getDescriptorSetLayout(s);
			if (!set || !setLayout || !set->cpuBuffer) { continue; }

			for (const auto& binding : setLayout->bindings()) {
				if (binding.descriptorType != VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR ||
					binding.cpuLayout != MVKDescriptorCPULayout::OneID ||
					!binding.descriptorCount) {
					continue;
				}

				auto* desc = reinterpret_cast<id<MTLAccelerationStructure>*>(set->cpuBuffer + binding.cpuOffset);
				MVKAccelerationStructure* mvkAS = MVKAccelerationStructure::getMVKAccelerationStructure(desc[0]);
					if (mvkAS) {
					instanceSBTOffsetBuffer = mvkAS->getInstanceShaderBindingTableOffsetBuffer();

					// Mark the TLAS and all referenced BLASes for GPU access.
					// Without useResource, Metal may evict the AS data after the first frame.
					id<MTLAccelerationStructure> tlasAS = desc[0];
					if (tlasAS) {
						[mtlEncoder useResource: tlasAS usage: MTLResourceUsageRead];
					}
					for (auto* as : cmdEncoder->getDevice()->getAccelerationStructures()) {
						if ((as->getType() == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ||
							 as->getType() == VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR) &&
							as->getMTLAccelerationStructure()) {
							[mtlEncoder useResource: as->getMTLAccelerationStructure() usage: MTLResourceUsageRead];
						}
					}

					break;
				}
			}

			if (instanceSBTOffsetBuffer) { break; }
		}
	}
	[mtlEncoder setBuffer: instanceSBTOffsetBuffer
				   offset: 0
				  atIndex: MVKRayTracingPipeline::kInstanceSBTOffsetBufferIndex];

	// Bind the intersection function table if present (for AABB geometry).
	if (rtPipeline->usesIntersectionFunctionTable()) {
		rtPipeline->updateMTLIntersectionFunctionTable(hitGroupIndices);
		id<MTLIntersectionFunctionTable> ift = rtPipeline->getMTLIntersectionFunctionTable();
		if (ift) {
			[mtlEncoder setIntersectionFunctionTable: ift
									 atBufferIndex: MVKRayTracingPipeline::kIntersectionFunctionTableBufferIndex];

			// Bind descriptor set resources to the intersection function table
			// so the intersection function can access images/buffers.
			if (vk._layout) {
				for (uint32_t s = 0; s < vk._layout->getDescriptorSetCount(); s++) {
					MVKDescriptorSet* set = vk._descriptorSets[s];
					if (set && set->gpuBufferObject) {
						[ift setBuffer: set->gpuBufferObject
							   offset: set->gpuBufferOffset
							  atIndex: s];
					}
				}
			}
		}
	}

	if (rtPipeline->needsMissShaderBindingTable()) {
		mvkBindShaderBindingTable(cmdEncoder, mtlEncoder,
								  MVKRayTracingPipeline::kMissSBTBufferIndex,
								  mvkDecodeShaderBindingTable(cmdEncoder->getDevice(), _missSBT));
	}
	if (rtPipeline->needsHitShaderBindingTable()) {
		mvkBindShaderBindingTable(cmdEncoder, mtlEncoder,
								  MVKRayTracingPipeline::kHitSBTBufferIndex,
								  hitGroupIndices);
	}
	if (rtPipeline->needsCallableShaderBindingTable()) {
		mvkBindShaderBindingTable(cmdEncoder, mtlEncoder,
								  MVKRayTracingPipeline::kCallableSBTBufferIndex,
								  mvkDecodeShaderBindingTable(cmdEncoder->getDevice(), _callableSBT));
	}

	// Dispatch one thread per ray (width x height x depth).
	// Use dispatchThreads so threads_per_grid is available for gl_LaunchSizeEXT.
	MTLSize gridSize = MTLSizeMake(_width, _height, _depth);

	MTLSize threadgroupSize;
	if (_depth <= 1) {
		threadgroupSize = MTLSizeMake(8, 8, 1);
	} else {
		threadgroupSize = MTLSizeMake(4, 4, 4);
	}

	[mtlEncoder dispatchThreads: gridSize
		  threadsPerThreadgroup: threadgroupSize];
}


#pragma mark -
#pragma mark MVKCmdBindRayTracingPipeline

VkResult MVKCmdBindRayTracingPipeline::setContent(MVKCommandBuffer* cmdBuff, VkPipeline pipeline) {
	_pipeline = pipeline;
	return VK_SUCCESS;
}

void MVKCmdBindRayTracingPipeline::encode(MVKCommandEncoder* cmdEncoder) {
	auto* rtPipeline = (MVKRayTracingPipeline*)_pipeline;
	cmdEncoder->setRayTracingPipeline(rtPipeline);
	// Set the compute layout so descriptor sets bound with VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR
	// get encoded properly by finalizeDispatchState().
	cmdEncoder->getState().setComputeLayout(rtPipeline->getLayout());
}
