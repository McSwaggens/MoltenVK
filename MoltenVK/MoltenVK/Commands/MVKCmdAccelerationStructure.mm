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
#include "MVKCommandPool.h"
#include "MVKAccelerationStructure.h"
#include "MVKQueryPool.h"

using namespace std;


#pragma mark -
#pragma mark Support functions

// Must match the instance conversion shader.
typedef struct {
	uint64_t instances;
	uint64_t instanceSBTOffsets;
	uint32_t arrayOfPointers;
} MVKAccelerationStructureInstanceParams;

// Dispatches one thread for each of the specified number of elements.
static void dispatchThreads(id<MTLComputeCommandEncoder> mtlComputeEnc, id<MTLComputePipelineState> mtlPSO, NSUInteger count) {
	[mtlComputeEnc dispatchThreads: MTLSizeMake(count, 1, 1)
			 threadsPerThreadgroup: MTLSizeMake(min(count, mtlPSO.maxTotalThreadsPerThreadgroup), 1, 1)];
}

static bool hasTransform(const VkAccelerationStructureGeometryKHR& geometry, const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
	return (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
			geometry.geometry.triangles.transformData.deviceAddress &&
			rangeInfo.primitiveCount);
}

// Top-level acceleration structure builds read the bottom-level acceleration structures referenced by
// their instances, which Metal cannot track, so make all acceleration structures resident.
static void useAccelerationStructures(MVKCommandEncoder* cmdEncoder, id<MTLAccelerationStructureCommandEncoder> mtlASEnc) {
	MVKUseResourceHelper rez;
	cmdEncoder->getDevice()->encodeAccelerationStructures(rez, MVKResourceUsageStages::Compute);
	auto& mtlRezs = rez.entries[MVKResourceUsageStages::Compute].read;
	if ( !mtlRezs.empty() ) {
		[mtlASEnc useResources: mtlRezs.data() count: mtlRezs.size() usage: MTLResourceUsageRead];
	}
}


#pragma mark -
#pragma mark MVKCmdBuildAccelerationStructures

VkResult MVKCmdBuildAccelerationStructures::setContent(MVKCommandBuffer* cmdBuff,
													   uint32_t infoCount,
													   const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
													   const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos) {
	// Copy the geometries and build ranges, and clear any pointers into app memory.
	_buildInfos.clear();
	_geometries.clear();
	_rangeInfos.clear();
	for (uint32_t infoIdx = 0; infoIdx < infoCount; infoIdx++) {
		const auto& buildInfo = pInfos[infoIdx];
		for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
			auto& geometry = _geometries.emplace_back(MVKAccelerationStructure::getGeometry(buildInfo, geoIdx));
			geometry.pNext = nullptr;
			geometry.geometry.triangles.pNext = nullptr;	// The geometry data union members share their pNext location
			_rangeInfos.push_back(ppBuildRangeInfos[infoIdx][geoIdx]);
		}
		auto& info = _buildInfos.emplace_back(buildInfo);
		info.pNext = nullptr;
		info.ppGeometries = nullptr;
	}

	// Now that the geometries will no longer move, point each build info at its geometries.
	const VkAccelerationStructureGeometryKHR* pGeometries = _geometries.data();
	for (auto& info : _buildInfos) {
		info.pGeometries = pGeometries;
		pGeometries += info.geometryCount;
	}

	return VK_SUCCESS;
}

const VkAccelerationStructureBuildRangeInfoKHR* MVKCmdBuildAccelerationStructures::getRangeInfos(const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo) {
	return &_rangeInfos[buildInfo.pGeometries - _geometries.data()];
}

void MVKCmdBuildAccelerationStructures::encode(MVKCommandEncoder* cmdEncoder) {
	size_t infoCount = _buildInfos.size();
	MTLAccelerationStructureDescriptor* mtlDescs[infoCount];
	for (size_t infoIdx = 0; infoIdx < infoCount; infoIdx++) {
		const auto& buildInfo = _buildInfos[infoIdx];
		mtlDescs[infoIdx] = cmdEncoder->getDevice()->getMTLAccelerationStructureDescriptor(buildInfo, getRangeInfos(buildInfo), true);
		if ( !mtlDescs[infoIdx] ) {
			cmdEncoder->reportError(VK_ERROR_INVALID_DEVICE_ADDRESS_EXT, "vkCmdBuildAccelerationStructuresKHR(): Acceleration structure geometry data must be held in a buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.");
		}
	}
	encodeInputConversions(cmdEncoder, mtlDescs);
	encodeBuilds(cmdEncoder, mtlDescs);
}

// Metal requires the instances of top-level acceleration structures, and geometry transforms, in different layouts
// than Vulkan. Convert them on the GPU, because earlier GPU commands may write them, and they may be in private memory.
void MVKCmdBuildAccelerationStructures::encodeInputConversions(MVKCommandEncoder* cmdEncoder, MTLAccelerationStructureDescriptor* const* mtlDescs) {
	id<MTLComputeCommandEncoder> mtlComputeEnc = nil;
	for (size_t infoIdx = 0; infoIdx < _buildInfos.size(); infoIdx++) {
		if ( !mtlDescs[infoIdx] ) { continue; }

		const auto& buildInfo = _buildInfos[infoIdx];
		const auto* pRangeInfos = getRangeInfos(buildInfo);
		bool isTopLevel = buildInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		bool needsConversion = false;
		for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
			needsConversion |= isTopLevel ? pRangeInfos[geoIdx].primitiveCount : hasTransform(buildInfo.pGeometries[geoIdx], pRangeInfos[geoIdx]);
		}

		if (needsConversion && !mtlComputeEnc) {
			mtlComputeEnc = cmdEncoder->getMTLComputeEncoder(kMVKCommandUseBuildAccelerationStructures);

			// The conversions read their inputs, and the headers of bottom-level acceleration structures, through device addresses.
			MVKDevice* mvkDev = cmdEncoder->getDevice();
			if ( !mvkDev->hasResidencySet() ) {
				MVKUseResourceHelper& rez = cmdEncoder->getState().mtlShared()._useResource;
				mvkDev->encodeGPUAddressableBuffers(rez, MVKResourceUsageStages::Compute);
				mvkDev->encodeAccelerationStructures(rez, MVKResourceUsageStages::Compute);
				rez.bindAndResetCompute(mtlComputeEnc);
			}
		}

		if (isTopLevel) {
			encodeInstanceConversion(cmdEncoder, mtlComputeEnc, buildInfo, (MTLInstanceAccelerationStructureDescriptor*)mtlDescs[infoIdx]);
		} else if (needsConversion) {
			encodeTransformConversion(cmdEncoder, mtlComputeEnc, buildInfo, (MTLPrimitiveAccelerationStructureDescriptor*)mtlDescs[infoIdx]);
		}
	}
}

// Converts the Vulkan instances to Metal instance descriptors, and writes their SBT record offsets to a buffer of the
// destination acceleration structure, and the address of that buffer to the header of the destination.
void MVKCmdBuildAccelerationStructures::encodeInstanceConversion(MVKCommandEncoder* cmdEncoder,
																 id<MTLComputeCommandEncoder> mtlComputeEnc,
																 const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
																 MTLInstanceAccelerationStructureDescriptor* mtlInstDesc) {
	uint32_t instCnt = (uint32_t)mtlInstDesc.instanceCount;
	const MVKMTLBufferAllocation* mtlInstAlloc = cmdEncoder->getTempMTLBuffer(max(instCnt, 1u) * sizeof(MTLIndirectAccelerationStructureInstanceDescriptor), true);
	mtlInstDesc.instanceDescriptorBuffer = mtlInstAlloc->_mtlBuffer;
	mtlInstDesc.instanceDescriptorBufferOffset = mtlInstAlloc->_offset;
	if ( !instCnt ) { return; }

	auto* mvkAccStruct = (MVKAccelerationStructure*)buildInfo.dstAccelerationStructure;
	id<MTLBuffer> sbtOffsetsMTLBuff = mvkAccStruct->getInstanceSBTOffsetsMTLBuffer(instCnt);
	if ( !sbtOffsetsMTLBuff ) {
		cmdEncoder->reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkCmdBuildAccelerationStructuresKHR(): Could not allocate the instance data of an acceleration structure.");
		return;
	}

	const auto& instances = buildInfo.pGeometries[0].geometry.instances;
	MVKAccelerationStructureInstanceParams params = {
		.instances = instances.data.deviceAddress + getRangeInfos(buildInfo)[0].primitiveOffset,
		.instanceSBTOffsets = sbtOffsetsMTLBuff.gpuAddress,
		.arrayOfPointers = instances.arrayOfPointers,
	};

	id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdConvertAccelerationStructureInstancesMTLComputePipelineState();
	MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
	mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
	mtlState.bindStructBytes(mtlComputeEnc, &params, 0);
	mtlState.bindBuffer(mtlComputeEnc, mtlInstAlloc->_mtlBuffer, mtlInstAlloc->_offset, 1);
	mtlState.bindBuffer(mtlComputeEnc, sbtOffsetsMTLBuff, 0, 2);
	mtlState.bindBuffer(mtlComputeEnc, mvkAccStruct->getHeaderMTLBuffer(), mvkAccStruct->getHeaderOffset(), 3);
	dispatchThreads(mtlComputeEnc, mtlPSO, instCnt);
}

// Converts the geometry transforms to the Metal matrix layout, in a temporary buffer referenced by the geometry descriptors.
void MVKCmdBuildAccelerationStructures::encodeTransformConversion(MVKCommandEncoder* cmdEncoder,
																  id<MTLComputeCommandEncoder> mtlComputeEnc,
																  const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
																  MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc) {
	const auto* pRangeInfos = getRangeInfos(buildInfo);
	MVKSmallVector<uint64_t, 8> vkTransformAddrs;
	MVKSmallVector<MTLAccelerationStructureTriangleGeometryDescriptor*, 8> mtlTriDescs;
	for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
		const auto& geometry = buildInfo.pGeometries[geoIdx];
		if (hasTransform(geometry, pRangeInfos[geoIdx])) {
			vkTransformAddrs.push_back(geometry.geometry.triangles.transformData.deviceAddress + pRangeInfos[geoIdx].transformOffset);
			mtlTriDescs.push_back((MTLAccelerationStructureTriangleGeometryDescriptor*)mtlPrimDesc.geometryDescriptors[geoIdx]);
		}
	}

	size_t xfmCnt = vkTransformAddrs.size();
	const MVKMTLBufferAllocation* mtlXfmAlloc = cmdEncoder->getTempMTLBuffer(xfmCnt * sizeof(MTLPackedFloat4x3), true);
	for (size_t xfmIdx = 0; xfmIdx < xfmCnt; xfmIdx++) {
		mtlTriDescs[xfmIdx].transformationMatrixBuffer = mtlXfmAlloc->_mtlBuffer;
		mtlTriDescs[xfmIdx].transformationMatrixBufferOffset = mtlXfmAlloc->_offset + xfmIdx * sizeof(MTLPackedFloat4x3);
	}

	id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdConvertAccelerationStructureTransformsMTLComputePipelineState();
	MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
	mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
	cmdEncoder->setComputeBytes(mtlComputeEnc, vkTransformAddrs.data(), xfmCnt * sizeof(uint64_t), 0);
	mtlState.bindBuffer(mtlComputeEnc, mtlXfmAlloc->_mtlBuffer, mtlXfmAlloc->_offset, 1);
	dispatchThreads(mtlComputeEnc, mtlPSO, xfmCnt);
}

void MVKCmdBuildAccelerationStructures::encodeBuilds(MVKCommandEncoder* cmdEncoder, MTLAccelerationStructureDescriptor* const* mtlDescs) {
	MVKDevice* mvkDev = cmdEncoder->getDevice();
	id<MTLAccelerationStructureCommandEncoder> mtlASEnc = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseBuildAccelerationStructures);
	bool needsAccelerationStructures = !mvkDev->hasResidencySet();
	for (size_t infoIdx = 0; infoIdx < _buildInfos.size(); infoIdx++) {
		MTLAccelerationStructureDescriptor* mtlDesc = mtlDescs[infoIdx];
		if ( !mtlDesc ) { continue; }

		const auto& buildInfo = _buildInfos[infoIdx];
		if (needsAccelerationStructures && buildInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
			useAccelerationStructures(cmdEncoder, mtlASEnc);
			needsAccelerationStructures = false;
		}

		VkDeviceSize scratchOffset = 0;
		id<MTLBuffer> scratchMTLBuff = nil;
		if (buildInfo.scratchData.deviceAddress) {
			scratchMTLBuff = mvkDev->getMTLBufferForDeviceAddress(buildInfo.scratchData.deviceAddress, &scratchOffset);
		}

		auto* dstAccStruct = (MVKAccelerationStructure*)buildInfo.dstAccelerationStructure;
		if (buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
			// Metal refits in place if the destination is nil.
			auto* srcAccStruct = (MVKAccelerationStructure*)buildInfo.srcAccelerationStructure;
			[mtlASEnc refitAccelerationStructure: srcAccStruct->getMTLAccelerationStructure()
									  descriptor: mtlDesc
									 destination: srcAccStruct == dstAccStruct ? nil : dstAccStruct->getMTLAccelerationStructure()
								   scratchBuffer: scratchMTLBuff
							 scratchBufferOffset: scratchOffset];
		} else if (scratchMTLBuff) {
			[mtlASEnc buildAccelerationStructure: dstAccStruct->getMTLAccelerationStructure()
									  descriptor: mtlDesc
								   scratchBuffer: scratchMTLBuff
							 scratchBufferOffset: scratchOffset];
		} else {
			cmdEncoder->reportError(VK_ERROR_INVALID_DEVICE_ADDRESS_EXT, "vkCmdBuildAccelerationStructuresKHR(): Acceleration structure scratch data must be held in a buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.");
		}
	}
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
	auto* srcAccStruct = (MVKAccelerationStructure*)_src;
	auto* dstAccStruct = (MVKAccelerationStructure*)_dst;

	id<MTLAccelerationStructureCommandEncoder> mtlASEnc = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseCopyAccelerationStructure);
	if (_mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR) {
		[mtlASEnc copyAndCompactAccelerationStructure: srcAccStruct->getMTLAccelerationStructure()
							  toAccelerationStructure: dstAccStruct->getMTLAccelerationStructure()];
	} else {
		[mtlASEnc copyAccelerationStructure: srcAccStruct->getMTLAccelerationStructure()
					toAccelerationStructure: dstAccStruct->getMTLAccelerationStructure()];
	}

	// The instance SBT record offsets of a top-level acceleration structure are held outside the Metal acceleration
	// structure. Copy them to a buffer of the destination, and write the address of that buffer to its header.
	id<MTLBuffer> srcSBTOffsetsMTLBuff = srcAccStruct->getInstanceShaderBindingTableOffsetBuffer();
	if ( !srcSBTOffsetsMTLBuff ) { return; }

	NSUInteger sbtOffsetsLength = srcSBTOffsetsMTLBuff.length;
	id<MTLBuffer> dstSBTOffsetsMTLBuff = dstAccStruct->getInstanceSBTOffsetsMTLBuffer(uint32_t(sbtOffsetsLength / sizeof(uint32_t)));
	if ( !dstSBTOffsetsMTLBuff ) {
		cmdEncoder->reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkCmdCopyAccelerationStructureKHR(): Could not allocate the instance data of an acceleration structure.");
		return;
	}
	uint64_t dstSBTOffsetsAddr = dstSBTOffsetsMTLBuff.gpuAddress;
	const MVKMTLBufferAllocation* mtlAddrAlloc = cmdEncoder->copyToTempMTLBufferAllocation(&dstSBTOffsetsAddr, sizeof(dstSBTOffsetsAddr));

	id<MTLBlitCommandEncoder> mtlBlitEnc = cmdEncoder->getMTLBlitEncoder(kMVKCommandUseCopyAccelerationStructure);
	[mtlBlitEnc copyFromBuffer: srcSBTOffsetsMTLBuff
				  sourceOffset: 0
					  toBuffer: dstSBTOffsetsMTLBuff
			 destinationOffset: 0
						  size: sbtOffsetsLength];
	[mtlBlitEnc copyFromBuffer: mtlAddrAlloc->_mtlBuffer
				  sourceOffset: mtlAddrAlloc->_offset
					  toBuffer: dstAccStruct->getHeaderMTLBuffer()
			 destinationOffset: dstAccStruct->getHeaderOffset() + offsetof(MVKAccelerationStructureHeader, instanceSBTOffsets)
						  size: sizeof(dstSBTOffsetsAddr)];
}


#pragma mark -
#pragma mark MVKCmdWriteAccelerationStructuresProperties

VkResult MVKCmdWriteAccelerationStructuresProperties::setContent(MVKCommandBuffer* cmdBuff,
																 uint32_t accelerationStructureCount,
																 const VkAccelerationStructureKHR* pAccelerationStructures,
																 VkQueryType queryType,
																 VkQueryPool queryPool,
																 uint32_t firstQuery) {
	_accelerationStructures.assign(pAccelerationStructures, pAccelerationStructures + accelerationStructureCount);
	_queryType = queryType;
	_queryPool = queryPool;
	_firstQuery = firstQuery;
	return VK_SUCCESS;
}

void MVKCmdWriteAccelerationStructuresProperties::encode(MVKCommandEncoder* cmdEncoder) {
	auto* mvkQryPool = (MVKAccelerationStructureQueryPool*)_queryPool;
	uint32_t queryCount = (uint32_t)_accelerationStructures.size();
	id<MTLBuffer> mtlResultsBuff = mvkQryPool->getMTLQueryResultsBuffer();

	cmdEncoder->resetQueries(mvkQryPool, _firstQuery, queryCount);
	if (_queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR) {
		id<MTLAccelerationStructureCommandEncoder> mtlASEnc = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseWriteAccelerationStructuresProperties);
		for (uint32_t asIdx = 0; asIdx < queryCount; asIdx++) {
			auto* mvkAccStruct = (MVKAccelerationStructure*)_accelerationStructures[asIdx];
			[mtlASEnc writeCompactedAccelerationStructureSize: mvkAccStruct->getMTLAccelerationStructure()
													 toBuffer: mtlResultsBuff
													   offset: mvkQryPool->getQueryOffset(_firstQuery + asIdx)
												 sizeDataType: MTLDataTypeULong];
		}
	} else {
		// Metal provides no access to the contents of acceleration structures, so they cannot be serialized.
		id<MTLBlitCommandEncoder> mtlBlitEnc = cmdEncoder->getMTLBlitEncoder(kMVKCommandUseWriteAccelerationStructuresProperties);
		[mtlBlitEnc fillBuffer: mtlResultsBuff
						 range: NSMakeRange(mvkQryPool->getQueryOffset(_firstQuery), queryCount * kMVKQuerySlotSizeInBytes)
						 value: 0];
	}
	for (uint32_t query = _firstQuery; query < _firstQuery + queryCount; query++) {
		mvkQryPool->endQuery(query, cmdEncoder);
	}
}
