/*
 * MVKCmdRayTracing.mm
 *
 * Copyright (c) 2015-2026 The Brenwill Workshop Ltd. (http://www.brenwill.com)
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
#include "MVKCommandPool.h"
#include "MVKPipeline.h"


// Populates the shader binding table regions of the dispatch parameters.
// A region of size zero is unused, and the shader reads it as a table with no records.
static void setShaderBindingTables(MVKRayTracingDispatchParams& params,
								   const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
								   const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
								   const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
								   const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable) {
	auto getAddress = [](const VkStridedDeviceAddressRegionKHR* pRegion) { return pRegion->size ? pRegion->deviceAddress : 0; };
	params.raygenAddress = getAddress(pRaygenShaderBindingTable);
	params.raygenStride = pRaygenShaderBindingTable->stride;
	params.missAddress = getAddress(pMissShaderBindingTable);
	params.missStride = pMissShaderBindingTable->stride;
	params.hitAddress = getAddress(pHitShaderBindingTable);
	params.hitStride = pHitShaderBindingTable->stride;
	params.callableAddress = getAddress(pCallableShaderBindingTable);
	params.callableStride = pCallableShaderBindingTable->stride;
}

// Binds the dispatch parameters of the bound ray tracing pipeline, and returns the pipeline.
static MVKRayTracingPipeline* bindDispatchParams(MVKCommandEncoder* cmdEncoder,
												 id<MTLComputeCommandEncoder> mtlEncoder,
												 const MVKRayTracingDispatchParams& cmdParams) {
	MVKRayTracingPipeline* pipeline = cmdEncoder->getRayTracingPipeline();
	MVKRayTracingDispatchParams params = cmdParams;
	params.maxRecursionDepth = pipeline->getMaxRecursionDepth();
	cmdEncoder->setComputeBytes(mtlEncoder, &params, sizeof(params),
								pipeline->getStageResources().implicitBuffers.ids[MVKImplicitBuffer::RayTracingDispatchParams]);
	return pipeline;
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
	_params = {};
	setShaderBindingTables(_params, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable);
	_params.launchWidth = width;
	_params.launchHeight = height;
	_params.launchDepth = depth;
	return VK_SUCCESS;
}

void MVKCmdTraceRays::encode(MVKCommandEncoder* cmdEncoder) {
	if ( !_params.launchWidth || !_params.launchHeight || !_params.launchDepth ) { return; }

	cmdEncoder->finalizeRayTracingDispatchState();	// Ensure all updated state has been submitted to Metal
	id<MTLComputeCommandEncoder> mtlEncoder = cmdEncoder->getMTLComputeEncoder(kMVKCommandUseTraceRays);
	MVKRayTracingPipeline* pipeline = bindDispatchParams(cmdEncoder, mtlEncoder, _params);
	MTLSize launchSize = MTLSizeMake(_params.launchWidth, _params.launchHeight, _params.launchDepth);
	[mtlEncoder dispatchThreads: launchSize
		  threadsPerThreadgroup: pipeline->getThreadgroupSize(launchSize)];
}


#pragma mark -
#pragma mark MVKCmdTraceRaysIndirect

VkResult MVKCmdTraceRaysIndirect::setContent(MVKCommandBuffer* cmdBuff,
											 const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
											 const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
											 const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
											 const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
											 VkDeviceAddress indirectDeviceAddress) {
	_params = {};
	setShaderBindingTables(_params, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable);
	_params.indirectLaunchAddress = indirectDeviceAddress;

	// The launch size is converted to threadgroup counts by reading it through its Metal buffer.
	_mtlIndirectBuffer = cmdBuff->getDevice()->getMTLBufferForDeviceAddress(indirectDeviceAddress, &_mtlIndirectBufferOffset);
	if ( !_mtlIndirectBuffer ) {
		return cmdBuff->reportError(VK_ERROR_INITIALIZATION_FAILED, "vkCmdTraceRaysIndirectKHR(): The indirect device address 0x%llx is not in any buffer.", indirectDeviceAddress);
	}
	return VK_SUCCESS;
}

void MVKCmdTraceRaysIndirect::encode(MVKCommandEncoder* cmdEncoder) {
	MVKRayTracingPipeline* pipeline = cmdEncoder->getRayTracingPipeline();
	MTLSize tgSize = pipeline->getThreadgroupSize();
	id<MTLComputeCommandEncoder> mtlEncoder = cmdEncoder->getMTLComputeEncoder(kMVKCommandUseTraceRays);

	// Convert the launch size to the threadgroup counts of an indirect Metal dispatch.
	const MVKMTLBufferAllocation* tgCounts = cmdEncoder->getTempMTLBuffer(sizeof(MTLDispatchThreadgroupsIndirectArguments), true);
	uint32_t tgSizes[] = { (uint32_t)tgSize.width, (uint32_t)tgSize.height, (uint32_t)tgSize.depth };
	MVKMetalComputeCommandEncoderState& mtlCompute = cmdEncoder->getMtlCompute();
	mtlCompute.bindPipeline(mtlEncoder, cmdEncoder->getCommandEncodingPool()->getCmdTraceRaysIndirectConvertBuffersMTLComputePipelineState());
	mtlCompute.bindBuffer(mtlEncoder, _mtlIndirectBuffer, _mtlIndirectBufferOffset, 0);
	mtlCompute.bindBuffer(mtlEncoder, tgCounts->_mtlBuffer, tgCounts->_offset, 1);
	mtlCompute.bindBytes(mtlEncoder, tgSizes, sizeof(tgSizes), 2);
	[mtlEncoder dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

	// The pipeline reads the launch size, and skips the threads of partial threadgroups outside it.
	cmdEncoder->finalizeRayTracingDispatchState();
	bindDispatchParams(cmdEncoder, mtlEncoder, _params);
	[mtlEncoder dispatchThreadgroupsWithIndirectBuffer: tgCounts->_mtlBuffer
								  indirectBufferOffset: tgCounts->_offset
								 threadsPerThreadgroup: tgSize];
}
