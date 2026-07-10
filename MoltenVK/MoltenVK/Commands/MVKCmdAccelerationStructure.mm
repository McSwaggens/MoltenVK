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
#include "MVKQueryPool.h"
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
		dstAS->clearRetainedBuffers();

		auto makePrimitiveDataBuffer = [&](uint32_t geometryIndex,
										   uint32_t primitiveCount,
										   VkGeometryFlagsKHR geometryFlags) -> id<MTLBuffer> {
			if (primitiveCount == 0) { return nil; }

			NSUInteger byteCount = (NSUInteger)primitiveCount * sizeof(MVKRTPrimitiveData);
			id<MTLBuffer> primitiveDataBuffer = [cmdEncoder->getMTLDevice() newBufferWithLength: byteCount
			                                                                             options: MTLResourceStorageModeShared];
			if (!primitiveDataBuffer) { return nil; }

			auto* primitiveData = (MVKRTPrimitiveData*)primitiveDataBuffer.contents;
			for (uint32_t primitiveIndex = 0; primitiveIndex < primitiveCount; primitiveIndex++) {
				primitiveData[primitiveIndex] = { geometryIndex, primitiveIndex, geometryFlags };
			}

			dstAS->retainBuffer(primitiveDataBuffer);
			[primitiveDataBuffer release];
			return primitiveDataBuffer;
		};

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

					// Vertex data is addressed from the bound vertex buffer plus firstVertex.
					// primitiveOffset applies to the index stream, not the vertex stream.
					VkDeviceSize vtxOffset = 0;
					if (triData.vertexData.deviceAddress) {
						VkDeviceAddress vtxAddr = triData.vertexData.deviceAddress;
						triDesc.vertexBuffer = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(vtxAddr, &vtxOffset);
						triDesc.vertexBufferOffset = (NSUInteger)(vtxOffset + rangeInfo.firstVertex * triData.vertexStride);
					}
					triDesc.vertexStride = triData.vertexStride;
					triDesc.triangleCount = rangeInfo.primitiveCount;
					triDesc.intersectionFunctionTableOffset = g;

					// Index data — apply primitiveOffset from build range info
					if (triData.indexType != VK_INDEX_TYPE_NONE_KHR &&
						triData.indexData.deviceAddress) {
						VkDeviceSize idxOffset = 0;
						VkDeviceAddress idxAddr = triData.indexData.deviceAddress + rangeInfo.primitiveOffset;
						triDesc.indexBuffer = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(idxAddr, &idxOffset);
						triDesc.indexBufferOffset = (NSUInteger)idxOffset;
						triDesc.indexType = (triData.indexType == VK_INDEX_TYPE_UINT16)
							? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
					}

					if (auto primitiveDataBuffer = makePrimitiveDataBuffer(g, rangeInfo.primitiveCount, geom.flags)) {
						triDesc.primitiveDataBuffer = primitiveDataBuffer;
						triDesc.primitiveDataBufferOffset = 0;
						triDesc.primitiveDataStride = sizeof(MVKRTPrimitiveData);
						triDesc.primitiveDataElementSize = sizeof(MVKRTPrimitiveData);
					}
					triDesc.opaque = (geom.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0;
					[geomDescs addObject: triDesc];

				} else if (geom.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
					auto& aabbData = geom.geometry.aabbs;
					MTLAccelerationStructureBoundingBoxGeometryDescriptor* aabbDesc =
						[MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];

					// Apply primitiveOffset from build range info
					if (aabbData.data.deviceAddress) {
						VkDeviceSize aabbOffset = 0;
						VkDeviceAddress aabbAddr = aabbData.data.deviceAddress + rangeInfo.primitiveOffset;
						aabbDesc.boundingBoxBuffer = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(aabbAddr, &aabbOffset);
						aabbDesc.boundingBoxBufferOffset = (NSUInteger)aabbOffset;
					}
					aabbDesc.boundingBoxStride = aabbData.stride;
					aabbDesc.boundingBoxCount = rangeInfo.primitiveCount;
					aabbDesc.intersectionFunctionTableOffset = g;
					if (auto primitiveDataBuffer = makePrimitiveDataBuffer(g, rangeInfo.primitiveCount, geom.flags)) {
						aabbDesc.primitiveDataBuffer = primitiveDataBuffer;
						aabbDesc.primitiveDataBufferOffset = 0;
						aabbDesc.primitiveDataStride = sizeof(MVKRTPrimitiveData);
						aabbDesc.primitiveDataElementSize = sizeof(MVKRTPrimitiveData);
					}
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
			if (@available(macOS 12.0, iOS 15.0, tvOS 16.0, *)) {
				primDesc.usage |= MTLAccelerationStructureUsageExtendedLimits;
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

			// Build a compact array containing only the BLASes referenced by this TLAS.
			NSMutableArray<id<MTLAccelerationStructure>>* blasArray = [NSMutableArray array];
			std::unordered_map<uint64_t, uint32_t> blasAddressToIndex;
			// Convert Vulkan instance descriptors to Metal format.
			// Vulkan: VkAccelerationStructureInstanceKHR (64 bytes each)
			//   [0-47]  transform (3x4 row-major float)
			//   [48]    instanceCustomIndex:24 + mask:8
			//   [52]    instanceShaderBindingTableRecordOffset:24 + flags:8
			//   [56]    accelerationStructureReference (uint64_t device address)
			// Metal: MTLAccelerationStructureUserIDInstanceDescriptor (68 bytes each)
			//   Same as MTLAccelerationStructureInstanceDescriptor but adds userID field
			//   for VkAccelerationStructureInstanceKHR::instanceCustomIndex mapping.
			if (geomCount > 0 && bi.geometries[0].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
				auto& instData = bi.geometries[0].geometry.instances;
				auto& rangeInfo = bi.buildRangeInfos[0];
				uint32_t instanceCount = rangeInfo.primitiveCount;
				instDesc.instanceCount = instanceCount;
				instDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
				instDesc.instanceDescriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);

				if (instData.data.deviceAddress && instanceCount > 0) {
					// Allocate a command-scoped temporary Metal buffer for converted descriptors.
					NSUInteger mtlInstSize = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor) * instanceCount;
					const MVKMTLBufferAllocation* mtlInstAlloc = cmdEncoder->getTempMTLBuffer(mtlInstSize);
					id<MTLBuffer> mtlInstBuf = mtlInstAlloc->_mtlBuffer;
					NSUInteger mtlInstOffset = mtlInstAlloc->_offset;
					id<MTLBuffer> sbtOffsetBuffer = [cmdEncoder->getMTLDevice() newBufferWithLength: sizeof(uint32_t) * instanceCount
					                                                                          options: MTLResourceStorageModeShared];
					uint32_t* sbtOffsets = sbtOffsetBuffer ? (uint32_t*)sbtOffsetBuffer.contents : nullptr;
					id<MTLBuffer> instanceFlagsBuffer = [cmdEncoder->getMTLDevice() newBufferWithLength: sizeof(uint32_t) * instanceCount
					                                                                            options: MTLResourceStorageModeShared];
					uint32_t* instanceFlags = instanceFlagsBuffer ? (uint32_t*)instanceFlagsBuffer.contents : nullptr;

					auto* mtlInsts = (MTLAccelerationStructureUserIDInstanceDescriptor*)((uint8_t*)mtlInstAlloc->getContents());

					// For non-arrayOfPointers, hoist the invariant buffer lookup out of the loop.
					const uint8_t* flatSrcData = nullptr;
					if (!instData.arrayOfPointers) {
						VkDeviceSize srcOffset = 0;
						id<MTLBuffer> srcBuf = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(instData.data.deviceAddress, &srcOffset);
						flatSrcData = (const uint8_t*)[srcBuf contents] + srcOffset;
					}

					for (uint32_t j = 0; j < instanceCount; j++) {
						const VkAccelerationStructureInstanceKHR* vkInst;
						if (instData.arrayOfPointers) {
							// data points to an array of device addresses, each pointing to an instance descriptor
							VkDeviceSize ptrOffset = 0;
							id<MTLBuffer> ptrBuf = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(
								instData.data.deviceAddress + j * sizeof(VkDeviceAddress), &ptrOffset);
							const VkDeviceAddress* instAddr = (const VkDeviceAddress*)((const uint8_t*)[ptrBuf contents] + ptrOffset);
							VkDeviceSize instOffset = 0;
							id<MTLBuffer> instBuf = cmdEncoder->getDevice()->getMTLBufferForDeviceAddress(*instAddr, &instOffset);
							vkInst = (const VkAccelerationStructureInstanceKHR*)((const uint8_t*)[instBuf contents] + instOffset);
						} else {
							vkInst = (const VkAccelerationStructureInstanceKHR*)(flatSrcData + j * sizeof(VkAccelerationStructureInstanceKHR));
						}

						// Convert VkTransformMatrixKHR (3x4 row-major) to MTLPackedFloat4x3 (4x3 column-major).
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
						mtlInsts[j].intersectionFunctionTableOffset = vkInst->instanceShaderBindingTableRecordOffset;
						if (sbtOffsets) {
							sbtOffsets[j] = vkInst->instanceShaderBindingTableRecordOffset;
						}
						if (instanceFlags) {
							instanceFlags[j] = vkInst->flags;
						}

						// Map instanceCustomIndex to Metal's userID
						mtlInsts[j].userID = vkInst->instanceCustomIndex;

						// Map the BLAS address directly to its wrapper, and add each referenced
						// Metal acceleration structure once to the TLAS descriptor.
						auto it = blasAddressToIndex.find(vkInst->accelerationStructureReference);
						uint32_t blasIdx = 0;
						if (it != blasAddressToIndex.end()) {
							blasIdx = it->second;
						} else if (auto* blas = MVKAccelerationStructure::getMVKAccelerationStructure(vkInst->accelerationStructureReference)) {
							auto type = blas->getType();
							id<MTLAccelerationStructure> mtlBLAS = blas->getMTLAccelerationStructure();
							if ((type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ||
								 type == VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR) && mtlBLAS) {
								blasIdx = (uint32_t)blasArray.count;
								blasAddressToIndex.emplace(vkInst->accelerationStructureReference, blasIdx);
								[blasArray addObject: mtlBLAS];
							}
						}
						mtlInsts[j].accelerationStructureIndex = blasIdx;
					}

					instDesc.instancedAccelerationStructures = blasArray;
					instDesc.instanceDescriptorBuffer = mtlInstBuf;
					instDesc.instanceDescriptorBufferOffset = mtlInstOffset;
					dstAS->setInstanceShaderBindingTableOffsetBuffer(sbtOffsetBuffer);
					dstAS->setInstanceFlagsBuffer(instanceFlagsBuffer);
					dstAS->setReferencedBLASes(blasArray);
					[sbtOffsetBuffer release];
					[instanceFlagsBuffer release];
				}
			}

			// Build flags
			if (bi.geometryInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR) {
				instDesc.usage |= MTLAccelerationStructureUsagePreferFastBuild;
			}
			if (bi.geometryInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) {
				instDesc.usage |= MTLAccelerationStructureUsageRefit;
			}
			if (@available(macOS 12.0, iOS 15.0, tvOS 16.0, *)) {
				instDesc.usage |= MTLAccelerationStructureUsageExtendedLimits;
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
	dstAS->copyRetainedBuffersFrom(srcAS);

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
	if (_queryType != VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR || !_queryPool) { return; }

	auto* mvkQueryPool = reinterpret_cast<MVKQueryPool*>(_queryPool);
	auto* asQueryPool = dynamic_cast<MVKAccelerationStructureQueryPool*>(mvkQueryPool);
	if (!asQueryPool || _accelerationStructures.empty()) { return; }

	cmdEncoder->resetQueries(mvkQueryPool, _firstQuery, (uint32_t)_accelerationStructures.size());

	id<MTLCommandBuffer> mtlCmdBuff = cmdEncoder->_mtlCmdBuffer;
	id<MTLAccelerationStructureCommandEncoder> asEncoder = [mtlCmdBuff accelerationStructureCommandEncoder];
	for (uint32_t i = 0; i < _accelerationStructures.size(); i++) {
		auto* mvkAS = (MVKAccelerationStructure*)_accelerationStructures[i];
		if (mvkAS && mvkAS->getMTLAccelerationStructure()) {
			NSUInteger offset = asQueryPool->getQueryOffset(_firstQuery + i);
			[asEncoder writeCompactedAccelerationStructureSize: mvkAS->getMTLAccelerationStructure()
													  toBuffer: asQueryPool->getMTLQueryResultsBuffer()
														offset: offset];
		}
		mvkQueryPool->endQuery(_firstQuery + i, cmdEncoder);
	}
	[asEncoder endEncoding];
}
