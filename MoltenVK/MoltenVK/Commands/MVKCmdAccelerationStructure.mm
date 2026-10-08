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
	uint32_t instanceCount;
	uint32_t arrayOfPointers;
} MVKAccelerationStructureInstanceParams;

// Must match the instance data copy shader.
typedef struct {
	uint64_t dstInstanceSBTOffsets;
	uint32_t dstCapacity;
} MVKAccelerationStructureInstanceCopyParams;

// Must match the vertex copy shader.
typedef struct {
	uint64_t vertices;
	uint32_t stride;
	uint32_t vertexSize;
} MVKAccelerationStructureVertexParams;

// Must match the bounding box conversion shader.
typedef struct {
	uint64_t boundingBoxes;
	uint64_t stride;
} MVKAccelerationStructureBoundingBoxParams;

// Temporary buffers larger than this are allocated for the Metal command buffer alone. Smaller ones come from the
// power-of-two pools of the command pool, which retain their MTLBuffers until the command pool is destroyed.
static constexpr NSUInteger kMVKMaxPooledTempMTLBufferLength = 256 * KIBI;

// Returns a private temporary buffer, which is released when the Metal command buffer completes, or nil if
// it cannot be allocated. The MTLBuffer is returned, and the offset within it via the pOffset parameter.
static id<MTLBuffer> getTempPrivateMTLBuffer(MVKCommandEncoder* cmdEncoder, NSUInteger length, NSUInteger* pOffset) {
	if (length <= kMVKMaxPooledTempMTLBufferLength) {
		const MVKMTLBufferAllocation* mtlBuffAlloc = cmdEncoder->getTempMTLBuffer(length, true);
		*pOffset = mtlBuffAlloc->_offset;
		return mtlBuffAlloc->_mtlBuffer;
	}

	*pOffset = 0;
	id<MTLBuffer> mtlBuff = [cmdEncoder->getMTLDevice() newBufferWithLength: length options: MTLResourceStorageModePrivate];	// retained
	if ( !mtlBuff ) {
		cmdEncoder->reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "Could not allocate a temporary buffer of %lu bytes for acceleration structure data.", (unsigned long)length);
		return nil;
	}
	[cmdEncoder->_mtlCmdBuffer addCompletedHandler: ^(id<MTLCommandBuffer> mcb) { [mtlBuff release]; }];
	return mtlBuff;
}

// Dispatches one thread for each of the specified number of elements.
static void dispatchThreads(id<MTLComputeCommandEncoder> mtlComputeEnc, id<MTLComputePipelineState> mtlPSO, NSUInteger count) {
	[mtlComputeEnc dispatchThreads: MTLSizeMake(count, 1, 1)
			 threadsPerThreadgroup: MTLSizeMake(min(count, mtlPSO.maxTotalThreadsPerThreadgroup), 1, 1)];
}

// Metal reads row-major transforms where available. Otherwise, they are converted to the Metal column-major layout.
static bool needsTransformConversion(const VkAccelerationStructureGeometryKHR& geometry,
									 const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo,
									 MTLAccelerationStructureGeometryDescriptor* mtlGeoDesc) {
	return (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
			geometry.geometry.triangles.transformData.deviceAddress &&
			rangeInfo.primitiveCount &&
			!((MTLAccelerationStructureTriangleGeometryDescriptor*)mtlGeoDesc).transformationMatrixBuffer);
}

// Metal requires vertex buffer offsets to be 4-byte aligned, while Vulkan only requires vertex component alignment.
static bool hasMisalignedVertices(const VkAccelerationStructureGeometryKHR& geometry, MTLAccelerationStructureGeometryDescriptor* mtlGeoDesc) {
	return (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
			((MTLAccelerationStructureTriangleGeometryDescriptor*)mtlGeoDesc).vertexBufferOffset % 4);
}

static bool hasBoundingBoxes(const VkAccelerationStructureGeometryKHR& geometry, const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
	return geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR && rangeInfo.primitiveCount;
}

// Returns whether any build inputs must be converted for Metal.
static bool needsInputConversion(const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
								 const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos,
								 MTLAccelerationStructureDescriptor* mtlDesc) {
	if (buildInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
		return ((MTLInstanceAccelerationStructureDescriptor*)mtlDesc).instanceCount;
	}

	NSArray<MTLAccelerationStructureGeometryDescriptor*>* mtlGeoDescs = ((MTLPrimitiveAccelerationStructureDescriptor*)mtlDesc).geometryDescriptors;
	for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
		const auto& geometry = buildInfo.pGeometries[geoIdx];
		if (needsTransformConversion(geometry, pRangeInfos[geoIdx], mtlGeoDescs[geoIdx]) ||
			hasMisalignedVertices(geometry, mtlGeoDescs[geoIdx]) ||
			hasBoundingBoxes(geometry, pRangeInfos[geoIdx])) { return true; }
	}
	return false;
}

// Returns a compute encoder for converting build inputs or copying instance data. The shaders read data through
// device addresses, including the headers and instance data of acceleration structures. Without a residency set,
// make these resident, once per Metal encoder, like for other shaders that access them.
static id<MTLComputeCommandEncoder> getAccelerationStructureMTLComputeEncoder(MVKCommandEncoder* cmdEncoder,
																			  MVKCommandUse cmdUse,
																			  bool needsGPUAddressableBuffers) {
	id<MTLComputeCommandEncoder> mtlComputeEnc = cmdEncoder->getMTLComputeEncoder(cmdUse);
	MVKDevice* mvkDev = cmdEncoder->getDevice();
	if ( !mvkDev->hasResidencySet() ) {
		MVKMetalSharedCommandEncoderState& mtlShared = cmdEncoder->getState().mtlShared();
		if (needsGPUAddressableBuffers && mtlShared._gpuAddressableResourceStages == MVKResourceUsageStages::None) {
			mtlShared._gpuAddressableResourceStages = MVKResourceUsageStages::Compute;
			mvkDev->encodeGPUAddressableBuffers(mtlShared._useResource, MVKResourceUsageStages::Compute);
			mtlShared._useResource.bindAndResetCompute(mtlComputeEnc);
		}
		if (mtlShared._accelerationStructureStages == MVKResourceUsageStages::None) {
			mtlShared._accelerationStructureStages = MVKResourceUsageStages::Compute;
			mvkDev->getAccelerationStructureHeaderPool()->useResources(mtlComputeEnc);
		}
	}
	return mtlComputeEnc;
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
	MVKSmallVector<MTLAccelerationStructureDescriptor*, 4> mtlDescs;
	mtlDescs.reserve(infoCount);
	for (const auto& buildInfo : _buildInfos) {
		MTLAccelerationStructureDescriptor* mtlDesc = cmdEncoder->getDevice()->getMTLAccelerationStructureDescriptor(buildInfo, getRangeInfos(buildInfo), true);
		if ( !mtlDesc ) {
			cmdEncoder->reportError(VK_ERROR_INVALID_DEVICE_ADDRESS_EXT, "vkCmdBuildAccelerationStructuresKHR(): Acceleration structure geometry data must be held in a buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.");
		}
		mtlDescs.push_back(mtlDesc);
	}
	encodeInputConversions(cmdEncoder, mtlDescs.data());
	encodeBuilds(cmdEncoder, mtlDescs.data());
}

// Some build inputs must be converted for Metal: the instances of top-level acceleration structures, and geometry
// transforms, have different layouts than in Vulkan, vertex offsets must be aligned, and bounding boxes are enlarged.
// Convert them on the GPU, because earlier GPU commands may write them, and they may be in private memory.
// The descriptor of a build whose inputs cannot be converted is cleared, so the build is skipped.
void MVKCmdBuildAccelerationStructures::encodeInputConversions(MVKCommandEncoder* cmdEncoder, MTLAccelerationStructureDescriptor** mtlDescs) {
	id<MTLComputeCommandEncoder> mtlComputeEnc = nil;
	for (size_t infoIdx = 0; infoIdx < _buildInfos.size(); infoIdx++) {
		MTLAccelerationStructureDescriptor* mtlDesc = mtlDescs[infoIdx];
		const auto& buildInfo = _buildInfos[infoIdx];
		if ( !mtlDesc || !needsInputConversion(buildInfo, getRangeInfos(buildInfo), mtlDesc) ) { continue; }

		if ( !mtlComputeEnc ) { mtlComputeEnc = getAccelerationStructureMTLComputeEncoder(cmdEncoder, kMVKCommandUseBuildAccelerationStructures, true); }

		bool wasConverted;
		if (buildInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
			wasConverted = encodeInstanceConversion(cmdEncoder, mtlComputeEnc, buildInfo, (MTLInstanceAccelerationStructureDescriptor*)mtlDesc);
		} else {
			auto* mtlPrimDesc = (MTLPrimitiveAccelerationStructureDescriptor*)mtlDesc;
			wasConverted = (encodeTransformConversion(cmdEncoder, mtlComputeEnc, buildInfo, mtlPrimDesc) &&
							encodeVertexAlignment(cmdEncoder, mtlComputeEnc, buildInfo, mtlPrimDesc) &&
							encodeBoundingBoxConversion(cmdEncoder, mtlComputeEnc, buildInfo, mtlPrimDesc));
		}
		if ( !wasConverted ) { mtlDescs[infoIdx] = nil; }
	}
}

// Converts the Vulkan instances to Metal instance descriptors, and writes their SBT record offsets to a buffer of the
// destination acceleration structure, and the address and number of those to the header slot of the destination.
bool MVKCmdBuildAccelerationStructures::encodeInstanceConversion(MVKCommandEncoder* cmdEncoder,
																 id<MTLComputeCommandEncoder> mtlComputeEnc,
																 const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
																 MTLInstanceAccelerationStructureDescriptor* mtlInstDesc) {
	uint32_t instCnt = (uint32_t)mtlInstDesc.instanceCount;
	NSUInteger mtlInstOffset = 0;
	id<MTLBuffer> mtlInstBuff = getTempPrivateMTLBuffer(cmdEncoder, instCnt * sizeof(MTLIndirectAccelerationStructureInstanceDescriptor), &mtlInstOffset);
	if ( !mtlInstBuff ) { return false; }
	mtlInstDesc.instanceDescriptorBuffer = mtlInstBuff;
	mtlInstDesc.instanceDescriptorBufferOffset = mtlInstOffset;

	auto* mvkAccStruct = (MVKAccelerationStructure*)buildInfo.dstAccelerationStructure;
	id<MTLBuffer> sbtOffsetsMTLBuff = mvkAccStruct->getInstanceSBTOffsetsMTLBuffer(instCnt);
	if ( !sbtOffsetsMTLBuff ) {
		cmdEncoder->reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkCmdBuildAccelerationStructuresKHR(): Could not allocate the instance data of an acceleration structure.");
		return false;
	}

	const auto& instances = buildInfo.pGeometries[0].geometry.instances;
	MVKAccelerationStructureInstanceParams params = {
		.instances = instances.data.deviceAddress + getRangeInfos(buildInfo)[0].primitiveOffset,
		.instanceSBTOffsets = sbtOffsetsMTLBuff.gpuAddress,
		.instanceCount = instCnt,
		.arrayOfPointers = instances.arrayOfPointers,
	};

	id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdConvertAccelerationStructureInstancesMTLComputePipelineState();
	MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
	mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
	mtlState.bindStructBytes(mtlComputeEnc, &params, 0);
	mtlState.bindBuffer(mtlComputeEnc, mtlInstBuff, mtlInstOffset, 1);
	mtlState.bindBuffer(mtlComputeEnc, sbtOffsetsMTLBuff, 0, 2);
	mtlState.bindBuffer(mtlComputeEnc, mvkAccStruct->getHeaderMTLBuffer(), mvkAccStruct->getHeaderOffset(), 3);
	dispatchThreads(mtlComputeEnc, mtlPSO, instCnt);
	return true;
}

// Converts the geometry transforms to the Metal matrix layout, in a temporary buffer referenced by the geometry descriptors.
bool MVKCmdBuildAccelerationStructures::encodeTransformConversion(MVKCommandEncoder* cmdEncoder,
																  id<MTLComputeCommandEncoder> mtlComputeEnc,
																  const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
																  MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc) {
	const auto* pRangeInfos = getRangeInfos(buildInfo);
	MVKSmallVector<uint64_t, 8> vkTransformAddrs;
	MVKSmallVector<MTLAccelerationStructureTriangleGeometryDescriptor*, 8> mtlTriDescs;
	for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
		const auto& geometry = buildInfo.pGeometries[geoIdx];
		auto* mtlGeoDesc = mtlPrimDesc.geometryDescriptors[geoIdx];
		if (needsTransformConversion(geometry, pRangeInfos[geoIdx], mtlGeoDesc)) {
			vkTransformAddrs.push_back(geometry.geometry.triangles.transformData.deviceAddress + pRangeInfos[geoIdx].transformOffset);
			mtlTriDescs.push_back((MTLAccelerationStructureTriangleGeometryDescriptor*)mtlGeoDesc);
		}
	}

	size_t xfmCnt = vkTransformAddrs.size();
	if ( !xfmCnt ) { return true; }

	NSUInteger mtlXfmOffset = 0;
	id<MTLBuffer> mtlXfmBuff = getTempPrivateMTLBuffer(cmdEncoder, xfmCnt * sizeof(MTLPackedFloat4x3), &mtlXfmOffset);
	if ( !mtlXfmBuff ) { return false; }
	for (size_t xfmIdx = 0; xfmIdx < xfmCnt; xfmIdx++) {
		mtlTriDescs[xfmIdx].transformationMatrixBuffer = mtlXfmBuff;
		mtlTriDescs[xfmIdx].transformationMatrixBufferOffset = mtlXfmOffset + xfmIdx * sizeof(MTLPackedFloat4x3);
	}

	id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdConvertAccelerationStructureTransformsMTLComputePipelineState();
	MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
	mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
	cmdEncoder->setComputeBytes(mtlComputeEnc, vkTransformAddrs.data(), xfmCnt * sizeof(uint64_t), 0);
	mtlState.bindBuffer(mtlComputeEnc, mtlXfmBuff, mtlXfmOffset, 1);
	dispatchThreads(mtlComputeEnc, mtlPSO, xfmCnt);
	return true;
}

// Copies the vertices of geometries whose vertex buffer offsets Metal does not support to temporary buffers,
// and references those from the geometry descriptors instead.
bool MVKCmdBuildAccelerationStructures::encodeVertexAlignment(MVKCommandEncoder* cmdEncoder,
															  id<MTLComputeCommandEncoder> mtlComputeEnc,
															  const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
															  MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc) {
	const auto* pRangeInfos = getRangeInfos(buildInfo);
	for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
		const auto& geometry = buildInfo.pGeometries[geoIdx];
		auto* mtlTriDesc = (MTLAccelerationStructureTriangleGeometryDescriptor*)mtlPrimDesc.geometryDescriptors[geoIdx];
		if ( !hasMisalignedVertices(geometry, mtlTriDesc) ) { continue; }

		// Indexed triangles access the vertices from firstVertex to maxVertex.
		const auto& triangles = geometry.geometry.triangles;
		const auto& rangeInfo = pRangeInfos[geoIdx];
		uint32_t vtxCnt = (triangles.indexType == VK_INDEX_TYPE_NONE_KHR
						   ? rangeInfo.primitiveCount * 3
						   : triangles.maxVertex + 1 - min(rangeInfo.firstVertex, triangles.maxVertex + 1));
		MVKAccelerationStructureVertexParams params = {
			.vertices = mtlTriDesc.vertexBuffer.gpuAddress + mtlTriDesc.vertexBufferOffset,
			.stride = (uint32_t)triangles.vertexStride,
			.vertexSize = cmdEncoder->getPixelFormats()->getBytesPerBlock(triangles.vertexFormat),
		};
		NSUInteger mtlVtxOffset = 0;
		id<MTLBuffer> mtlVtxBuff = getTempPrivateMTLBuffer(cmdEncoder, max(vtxCnt, 1u) * triangles.vertexStride, &mtlVtxOffset);
		if ( !mtlVtxBuff ) { return false; }
		mtlTriDesc.vertexBuffer = mtlVtxBuff;
		mtlTriDesc.vertexBufferOffset = mtlVtxOffset;
		if ( !vtxCnt ) { continue; }

		id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdCopyAccelerationStructureVerticesMTLComputePipelineState();
		MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
		mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
		mtlState.bindStructBytes(mtlComputeEnc, &params, 0);
		mtlState.bindBuffer(mtlComputeEnc, mtlVtxBuff, mtlVtxOffset, 1);
		dispatchThreads(mtlComputeEnc, mtlPSO, vtxCnt);
	}
	return true;
}

// Metal does not report the intersection of a ray that runs exactly along a face of a bounding box, while Vulkan expects
// it. Bounding box intersections may be reported conservatively, so build from slightly enlarged copies of the boxes.
bool MVKCmdBuildAccelerationStructures::encodeBoundingBoxConversion(MVKCommandEncoder* cmdEncoder,
																	id<MTLComputeCommandEncoder> mtlComputeEnc,
																	const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
																	MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc) {
	const auto* pRangeInfos = getRangeInfos(buildInfo);
	for (uint32_t geoIdx = 0; geoIdx < buildInfo.geometryCount; geoIdx++) {
		const auto& geometry = buildInfo.pGeometries[geoIdx];
		const auto& rangeInfo = pRangeInfos[geoIdx];
		if ( !hasBoundingBoxes(geometry, rangeInfo) ) { continue; }

		auto* mtlBoxDesc = (MTLAccelerationStructureBoundingBoxGeometryDescriptor*)mtlPrimDesc.geometryDescriptors[geoIdx];
		MVKAccelerationStructureBoundingBoxParams params = {
			.boundingBoxes = mtlBoxDesc.boundingBoxBuffer.gpuAddress + mtlBoxDesc.boundingBoxBufferOffset,
			.stride = mtlBoxDesc.boundingBoxStride,
		};
		NSUInteger mtlBoxOffset = 0;
		id<MTLBuffer> mtlBoxBuff = getTempPrivateMTLBuffer(cmdEncoder, rangeInfo.primitiveCount * sizeof(MTLAxisAlignedBoundingBox), &mtlBoxOffset);
		if ( !mtlBoxBuff ) { return false; }
		mtlBoxDesc.boundingBoxBuffer = mtlBoxBuff;
		mtlBoxDesc.boundingBoxBufferOffset = mtlBoxOffset;
		mtlBoxDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);

		id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdConvertAccelerationStructureBoundingBoxesMTLComputePipelineState();
		MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
		mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
		mtlState.bindStructBytes(mtlComputeEnc, &params, 0);
		mtlState.bindBuffer(mtlComputeEnc, mtlBoxBuff, mtlBoxOffset, 1);
		dispatchThreads(mtlComputeEnc, mtlPSO, rangeInfo.primitiveCount);
	}
	return true;
}

void MVKCmdBuildAccelerationStructures::encodeBuilds(MVKCommandEncoder* cmdEncoder, MTLAccelerationStructureDescriptor* const* mtlDescs) {
	MVKDevice* mvkDev = cmdEncoder->getDevice();
	id<MTLAccelerationStructureCommandEncoder> mtlASEnc = nil;
	bool needsAccelerationStructures = !mvkDev->hasResidencySet();
	for (size_t infoIdx = 0; infoIdx < _buildInfos.size(); infoIdx++) {
		MTLAccelerationStructureDescriptor* mtlDesc = mtlDescs[infoIdx];
		if ( !mtlDesc ) { continue; }

		const auto& buildInfo = _buildInfos[infoIdx];
		VkDeviceSize scratchOffset = 0;
		id<MTLBuffer> scratchMTLBuff = nil;
		if (buildInfo.scratchData.deviceAddress) {
			scratchMTLBuff = mvkDev->getMTLBufferForDeviceAddress(buildInfo.scratchData.deviceAddress, &scratchOffset);
		}
		if ( !scratchMTLBuff ) {
			cmdEncoder->reportError(VK_ERROR_INVALID_DEVICE_ADDRESS_EXT, "vkCmdBuildAccelerationStructuresKHR(): Acceleration structure scratch data must be held in a buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.");
			continue;
		}

		if ( !mtlASEnc ) { mtlASEnc = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseBuildAccelerationStructures); }

		// Top-level acceleration structure builds read the bottom-level acceleration structures referenced by
		// their instances, which Metal cannot track, so make all acceleration structures resident.
		if (needsAccelerationStructures && buildInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
			mvkDev->getAccelerationStructureHeaderPool()->useMTLAccelerationStructures(mtlASEnc);
			needsAccelerationStructures = false;
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
		} else {
			[mtlASEnc buildAccelerationStructure: dstAccStruct->getMTLAccelerationStructure()
									  descriptor: mtlDesc
								   scratchBuffer: scratchMTLBuff
							 scratchBufferOffset: scratchOffset];
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

	// Bottom-level acceleration structures have no instance data.
	if (srcAccStruct->getType() != VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) { encodeInstanceDataCopy(cmdEncoder); }
}

// The instance SBT record offsets of a top-level acceleration structure are held outside the Metal acceleration structure,
// in a buffer whose address and size are in its header slot. These are written on the GPU, so copy the instance data on
// the GPU, where the header slot of the source reflects the commands that execute before the copy, to a buffer of the
// destination, and write the address and size of that buffer to the header slot of the destination. The buffer is sized
// for the largest instance data of the source that previously encoded commands may have written. If the source holds
// more instances when the copy executes, which is only possible if it is built by commands encoded after the copy, the
// destination shares the instance data of the source instead, which is retained until the source is destroyed.
void MVKCmdCopyAccelerationStructure::encodeInstanceDataCopy(MVKCommandEncoder* cmdEncoder) {
	auto* srcAccStruct = (MVKAccelerationStructure*)_src;
	auto* dstAccStruct = (MVKAccelerationStructure*)_dst;

	uint32_t capacity = max(srcAccStruct->getInstanceSBTOffsetsCapacity(), dstAccStruct->getInstanceSBTOffsetsCapacity());
	id<MTLBuffer> dstSBTOffsetsMTLBuff = dstAccStruct->getInstanceSBTOffsetsMTLBuffer(capacity);
	if ( !dstSBTOffsetsMTLBuff ) {
		cmdEncoder->reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkCmdCopyAccelerationStructureKHR(): Could not allocate the instance data of an acceleration structure.");
		return;
	}
	MVKAccelerationStructureInstanceCopyParams params = {
		.dstInstanceSBTOffsets = dstSBTOffsetsMTLBuff.gpuAddress,
		.dstCapacity = uint32_t(dstSBTOffsetsMTLBuff.length / sizeof(uint32_t)),
	};

	id<MTLComputeCommandEncoder> mtlComputeEnc = getAccelerationStructureMTLComputeEncoder(cmdEncoder, kMVKCommandUseCopyAccelerationStructure, false);
	id<MTLComputePipelineState> mtlPSO = cmdEncoder->getCommandEncodingPool()->getCmdCopyAccelerationStructureInstanceDataMTLComputePipelineState();
	MVKMetalComputeCommandEncoderState& mtlState = cmdEncoder->getMtlCompute();
	mtlState.bindPipeline(mtlComputeEnc, mtlPSO);
	mtlState.bindStructBytes(mtlComputeEnc, &params, 0);
	mtlState.bindBuffer(mtlComputeEnc, srcAccStruct->getHeaderMTLBuffer(), srcAccStruct->getHeaderOffset(), 1);
	mtlState.bindBuffer(mtlComputeEnc, dstAccStruct->getHeaderMTLBuffer(), dstAccStruct->getHeaderOffset(), 2);
	mtlState.bindBuffer(mtlComputeEnc, dstSBTOffsetsMTLBuff, 0, 3);
	dispatchThreads(mtlComputeEnc, mtlPSO, params.dstCapacity);
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
	_queryPool = queryPool;
	_firstQuery = firstQuery;
	return VK_SUCCESS;
}

void MVKCmdWriteAccelerationStructuresProperties::encode(MVKCommandEncoder* cmdEncoder) {
	auto* mvkQryPool = (MVKAccelerationStructureQueryPool*)_queryPool;
	uint32_t queryCount = (uint32_t)_accelerationStructures.size();
	id<MTLBuffer> mtlResultsBuff = mvkQryPool->getMTLQueryResultsBuffer();

	// Compacted size is the only supported acceleration structure query type.
	cmdEncoder->resetQueries(mvkQryPool, _firstQuery, queryCount);
	id<MTLAccelerationStructureCommandEncoder> mtlASEnc = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseWriteAccelerationStructuresProperties);
	for (uint32_t asIdx = 0; asIdx < queryCount; asIdx++) {
		auto* mvkAccStruct = (MVKAccelerationStructure*)_accelerationStructures[asIdx];
		[mtlASEnc writeCompactedAccelerationStructureSize: mvkAccStruct->getMTLAccelerationStructure()
												 toBuffer: mtlResultsBuff
												   offset: mvkQryPool->getQueryOffset(_firstQuery + asIdx)
											 sizeDataType: MTLDataTypeULong];
	}
	for (uint32_t query = _firstQuery; query < _firstQuery + queryCount; query++) {
		mvkQryPool->endQuery(query, cmdEncoder);
	}
}
