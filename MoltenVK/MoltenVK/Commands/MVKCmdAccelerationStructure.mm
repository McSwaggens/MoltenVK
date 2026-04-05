/*
 * MVKCmdAccelerationStructure.mm
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

#include "MVKCmdAccelerationStructure.h"
#include "MVKCommandBuffer.h"
#include "MVKAccelerationStructure.h"
#include "MVKBuffer.h"
#include "MVKCommandPool.h"
#include <unordered_map>

#pragma mark -
#pragma mark MVKCmdBuildAccelerationStructures

VkResult MVKCmdBuildAccelerationStructures::setContent(MVKCommandBuffer* cmdBuff,
                                                       uint32_t infoCount,
                                                       const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
                                                       const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos) {
	_buildInfos.resize(infoCount);
	for (uint32_t i = 0; i < infoCount; i++) {
		auto& bi = _buildInfos[i];
		bi.geometryInfo = pInfos[i];

		uint32_t geomCount = pInfos[i].geometryCount;
		bi.geometries.resize(geomCount);
		const VkAccelerationStructureGeometryKHR* pGeometries =
			pInfos[i].pGeometries ? pInfos[i].pGeometries : nullptr;
		const VkAccelerationStructureGeometryKHR* const* ppGeometries =
			pInfos[i].ppGeometries;

		for (uint32_t g = 0; g < geomCount; g++) {
			if (pGeometries) {
				bi.geometries[g] = pGeometries[g];
			} else if (ppGeometries) {
				bi.geometries[g] = *ppGeometries[g];
			}
		}

		bi.buildRangeInfos.resize(geomCount);
		for (uint32_t g = 0; g < geomCount; g++) {
			bi.buildRangeInfos[g] = ppBuildRangeInfos[i][g];
		}
	}
	return VK_SUCCESS;
}

void MVKCmdBuildAccelerationStructures::encode(MVKCommandEncoder* cmdEncoder) {
	id<MTLCommandBuffer> mtlCmdBuff = cmdEncoder->_mtlCmdBuffer;
	id<MTLAccelerationStructureCommandEncoder> asEncoder = [mtlCmdBuff accelerationStructureCommandEncoder];

	for (auto& bi : _buildInfos) {
		auto* dstAS = (MVKAccelerationStructure*)bi.geometryInfo.dstAccelerationStructure;
		id<MTLAccelerationStructure> mtlDstAS = dstAS->getMTLAccelerationStructure();
		uint32_t geomCount = bi.geometryInfo.geometryCount;

		if (bi.geometryInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) {
			// Build a primitive (bottom-level) acceleration structure.
			MTLPrimitiveAccelerationStructureDescriptor* primDesc =
				[MTLPrimitiveAccelerationStructureDescriptor descriptor];

			NSMutableArray* geomDescs = [NSMutableArray arrayWithCapacity: geomCount];
			for (uint32_t g = 0; g < geomCount; g++) {
				auto& geom = bi.geometries[g];
				auto& rangeInfo = bi.buildRangeInfos[g];

				if (geom.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
					auto& triData = geom.geometry.triangles;
					MTLAccelerationStructureTriangleGeometryDescriptor* triDesc =
						[MTLAccelerationStructureTriangleGeometryDescriptor descriptor];

					// Vertex data
					VkDeviceSize vtxOffset = 0;
					if (triData.vertexData.deviceAddress) {
						triDesc.vertexBuffer = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(triData.vertexData.deviceAddress, &vtxOffset);
						triDesc.vertexBufferOffset = (NSUInteger)vtxOffset;
						// Vertex buffer should be resolved from device address.
					}
					triDesc.vertexStride = triData.vertexStride;
					triDesc.triangleCount = rangeInfo.primitiveCount;

					// Index data
					if (triData.indexType != VK_INDEX_TYPE_NONE_KHR &&
						triData.indexData.deviceAddress) {
						VkDeviceSize idxOffset = 0;
						triDesc.indexBuffer = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(triData.indexData.deviceAddress, &idxOffset);
						triDesc.indexBufferOffset = (NSUInteger)idxOffset;
						triDesc.indexType = (triData.indexType == VK_INDEX_TYPE_UINT16)
							? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
					}

					triDesc.opaque = (geom.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0;
					[geomDescs addObject: triDesc];

				} else if (geom.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
					auto& aabbData = geom.geometry.aabbs;
					MTLAccelerationStructureBoundingBoxGeometryDescriptor* aabbDesc =
						[MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];

					if (aabbData.data.deviceAddress) {
						VkDeviceSize aabbOffset = 0;
						aabbDesc.boundingBoxBuffer = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(aabbData.data.deviceAddress, &aabbOffset);
						aabbDesc.boundingBoxBufferOffset = (NSUInteger)aabbOffset;
					}
					aabbDesc.boundingBoxStride = aabbData.stride;
					aabbDesc.boundingBoxCount = rangeInfo.primitiveCount;
					aabbDesc.opaque = (geom.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0;

					[geomDescs addObject: aabbDesc];
				}
			}

			primDesc.geometryDescriptors = geomDescs;

			// Build flags
			if (bi.geometryInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR) {
				primDesc.usage |= MTLAccelerationStructureUsagePreferFastBuild;
			}
			if (bi.geometryInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) {
				primDesc.usage |= MTLAccelerationStructureUsageRefit;
			}

			// Scratch buffer
			id<MTLBuffer> scratchBuf = nil;
			NSUInteger scratchOffset = 0;
			if (bi.geometryInfo.scratchData.deviceAddress) {
				VkDeviceSize scrOff = 0;
				scratchBuf = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(bi.geometryInfo.scratchData.deviceAddress, &scrOff);
				scratchOffset = (NSUInteger)scrOff;
			}

			if (bi.geometryInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
				[asEncoder buildAccelerationStructure: mtlDstAS
										  descriptor: primDesc
									   scratchBuffer: scratchBuf
								 scratchBufferOffset: scratchOffset];
			} else {
				// Update/refit
				id<MTLAccelerationStructure> mtlSrcAS = mtlDstAS;
				if (bi.geometryInfo.srcAccelerationStructure) {
					auto* srcAS = (MVKAccelerationStructure*)bi.geometryInfo.srcAccelerationStructure;
					mtlSrcAS = srcAS->getMTLAccelerationStructure();
				}
				[asEncoder refitAccelerationStructure: mtlSrcAS
										  descriptor: primDesc
										 destination: mtlDstAS
									   scratchBuffer: scratchBuf
								 scratchBufferOffset: scratchOffset];
			}
		} else {
			// Build an instance (top-level) acceleration structure.
			MTLInstanceAccelerationStructureDescriptor* instDesc =
				[MTLInstanceAccelerationStructureDescriptor descriptor];

			// Collect all known bottom-level acceleration structures.
			// Build a map from BLAS device address to index for instance descriptor conversion.
			auto& allAS = cmdEncoder->getDevice()->getAccelerationStructures();
			NSMutableArray<id<MTLAccelerationStructure>>* blasArray = [NSMutableArray array];
			std::unordered_map<uint64_t, uint32_t> blasAddressToIndex;
			for (auto* as : allAS) {
				if (as->getType() == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR &&
					as->getMTLAccelerationStructure()) {
					uint32_t idx = (uint32_t)[blasArray count];
					blasAddressToIndex[as->getDeviceAddress()] = idx;
					[blasArray addObject: as->getMTLAccelerationStructure()];
				}
			}
			instDesc.instancedAccelerationStructures = blasArray;

			// Convert Vulkan instance descriptors to Metal format.
			// Vulkan: VkAccelerationStructureInstanceKHR (64 bytes each)
			//   [0-47]  transform (3x4 row-major float)
			//   [48]    instanceCustomIndex:24 + mask:8
			//   [52]    instanceShaderBindingTableRecordOffset:24 + flags:8
			//   [56]    accelerationStructureReference (uint64_t device address)
			// Metal: MTLAccelerationStructureInstanceDescriptor (64 bytes each)
			//   [0-47]  transformationMatrix (4x3 column-major, same memory layout)
			//   [48]    options (uint32_t)
			//   [52]    mask (uint32_t)
			//   [56]    intersectionFunctionTableOffset (uint32_t)
			//   [60]    accelerationStructureIndex (uint32_t)
			if (geomCount > 0 && bi.geometries[0].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
				auto& instData = bi.geometries[0].geometry.instances;
				auto& rangeInfo = bi.buildRangeInfos[0];
				uint32_t instanceCount = rangeInfo.primitiveCount;
				instDesc.instanceCount = instanceCount;

				if (instData.data.deviceAddress && instanceCount > 0) {
					// Read the Vulkan instance descriptors from the source buffer
					VkDeviceSize srcOffset = 0;
					id<MTLBuffer> srcBuf = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(instData.data.deviceAddress, &srcOffset);
					const uint8_t* srcData = (const uint8_t*)[srcBuf contents] + srcOffset;

					// Allocate a temporary Metal buffer for converted descriptors
					NSUInteger mtlInstSize = sizeof(MTLAccelerationStructureInstanceDescriptor) * instanceCount;
					id<MTLBuffer> mtlInstBuf = [cmdEncoder->getDevice()->getPhysicalDevice()->getMTLDevice() newBufferWithLength: mtlInstSize
					                                                                   options: MTLResourceStorageModeShared];

					auto* mtlInsts = (MTLAccelerationStructureInstanceDescriptor*)[mtlInstBuf contents];
					for (uint32_t j = 0; j < instanceCount; j++) {
						const VkAccelerationStructureInstanceKHR* vkInst =
							(const VkAccelerationStructureInstanceKHR*)(srcData + j * sizeof(VkAccelerationStructureInstanceKHR));

						// Convert VkTransformMatrixKHR (3x4 row-major) to MTLPackedFloat4x3 (4x3 column-major).
						// VkTransformMatrixKHR: matrix[row][col], 3 rows x 4 cols
						// MTLPackedFloat4x3: columns[col][row], 4 cols x 3 rows
						for (int col = 0; col < 4; col++) {
							for (int row = 0; row < 3; row++) {
								mtlInsts[j].transformationMatrix.columns[col][row] = vkInst->transform.matrix[row][col];
							}
						}

						// Convert flags
						MTLAccelerationStructureInstanceOptions options = MTLAccelerationStructureInstanceOptionNone;
						if (vkInst->flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)
							options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
						if (vkInst->flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR)
							options |= MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
						if (vkInst->flags & VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR)
							options |= MTLAccelerationStructureInstanceOptionOpaque;
						if (vkInst->flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR)
							options |= MTLAccelerationStructureInstanceOptionNonOpaque;
						mtlInsts[j].options = options;

						mtlInsts[j].mask = vkInst->mask;
						mtlInsts[j].intersectionFunctionTableOffset = 0;

						// Map BLAS device address to index in instancedAccelerationStructures
						auto it = blasAddressToIndex.find(vkInst->accelerationStructureReference);
						mtlInsts[j].accelerationStructureIndex = (it != blasAddressToIndex.end()) ? it->second : 0;
					}

					instDesc.instanceDescriptorBuffer = mtlInstBuf;
					instDesc.instanceDescriptorBufferOffset = 0;
					// mtlInstBuf is retained by newBufferWithLength. The AS build
					// command encoder retains it during execution. We intentionally
					// don't release it here to avoid use-after-free.
				}
			}

			// Build flags
			if (bi.geometryInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR) {
				instDesc.usage |= MTLAccelerationStructureUsagePreferFastBuild;
			}
			if (bi.geometryInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) {
				instDesc.usage |= MTLAccelerationStructureUsageRefit;
			}

			// Scratch buffer
			id<MTLBuffer> scratchBuf = nil;
			NSUInteger scratchOffset = 0;
			if (bi.geometryInfo.scratchData.deviceAddress) {
				VkDeviceSize scrOff = 0;
				scratchBuf = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(bi.geometryInfo.scratchData.deviceAddress, &scrOff);
				scratchOffset = (NSUInteger)scrOff;
			}

			if (bi.geometryInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
				[asEncoder buildAccelerationStructure: mtlDstAS
										  descriptor: instDesc
									   scratchBuffer: scratchBuf
								 scratchBufferOffset: scratchOffset];
			} else {
				id<MTLAccelerationStructure> mtlSrcAS = mtlDstAS;
				if (bi.geometryInfo.srcAccelerationStructure) {
					auto* srcAS = (MVKAccelerationStructure*)bi.geometryInfo.srcAccelerationStructure;
					mtlSrcAS = srcAS->getMTLAccelerationStructure();
				}
				[asEncoder refitAccelerationStructure: mtlSrcAS
										  descriptor: instDesc
										 destination: mtlDstAS
									   scratchBuffer: scratchBuf
								 scratchBufferOffset: scratchOffset];
			}
			// mtlInstBuf is intentionally leaked (not released) to ensure it lives
			// until the GPU finishes executing the acceleration structure build.
		}
	}

	[asEncoder endEncoding];
}


#pragma mark -
#pragma mark MVKCmdCopyAccelerationStructure

VkResult MVKCmdCopyAccelerationStructure::setContent(MVKCommandBuffer* cmdBuff,
                                                     const VkCopyAccelerationStructureInfoKHR* pInfo) {
	_src = pInfo->src;
	_dst = pInfo->dst;
	_mode = pInfo->mode;
	return VK_SUCCESS;
}

void MVKCmdCopyAccelerationStructure::encode(MVKCommandEncoder* cmdEncoder) {
	auto* srcAS = (MVKAccelerationStructure*)_src;
	auto* dstAS = (MVKAccelerationStructure*)_dst;

	id<MTLCommandBuffer> mtlCmdBuff = cmdEncoder->_mtlCmdBuffer;
	id<MTLAccelerationStructureCommandEncoder> asEncoder = [mtlCmdBuff accelerationStructureCommandEncoder];

	if (_mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR) {
		[asEncoder copyAndCompactAccelerationStructure: srcAS->getMTLAccelerationStructure()
							   toAccelerationStructure: dstAS->getMTLAccelerationStructure()];
	} else {
		[asEncoder copyAccelerationStructure: srcAS->getMTLAccelerationStructure()
					 toAccelerationStructure: dstAS->getMTLAccelerationStructure()];
	}

	[asEncoder endEncoding];
}


#pragma mark -
#pragma mark MVKCmdWriteAccelerationStructuresProperties

VkResult MVKCmdWriteAccelerationStructuresProperties::setContent(MVKCommandBuffer* cmdBuff,
                                                                  uint32_t accelerationStructureCount,
                                                                  const VkAccelerationStructureKHR* pAccelerationStructures,
                                                                  VkQueryType queryType,
                                                                  VkQueryPool queryPool,
                                                                  uint32_t firstQuery) {
	_accelerationStructures.assign(pAccelerationStructures,
								   pAccelerationStructures + accelerationStructureCount);
	_queryType = queryType;
	_queryPool = queryPool;
	_firstQuery = firstQuery;
	return VK_SUCCESS;
}

void MVKCmdWriteAccelerationStructuresProperties::encode(MVKCommandEncoder* cmdEncoder) {
	// For compacted size queries, write to query pool buffer using AS command encoder.
	if (_queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR) {
		id<MTLCommandBuffer> mtlCmdBuff = cmdEncoder->_mtlCmdBuffer;
		id<MTLAccelerationStructureCommandEncoder> asEncoder = [mtlCmdBuff accelerationStructureCommandEncoder];

		// TODO: Get the query pool's MTLBuffer and write compacted sizes.
		// For now, just encode the compacted size queries individually.
		for (uint32_t i = 0; i < _accelerationStructures.size(); i++) {
			auto* mvkAS = (MVKAccelerationStructure*)_accelerationStructures[i];
			(void)mvkAS; // Placeholder — full implementation requires query pool buffer access
		}

		[asEncoder endEncoding];
	}
}
