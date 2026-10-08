/*
 * MVKCmdAccelerationStructure.h
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

#pragma once

#include "MVKCommand.h"
#include "MVKSmallVector.h"

#import <Metal/Metal.h>


#pragma mark -
#pragma mark MVKCmdBuildAccelerationStructures

/** Vulkan command to build or update acceleration structures. */
class MVKCmdBuildAccelerationStructures : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						uint32_t infoCount,
						const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
						const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;
	const VkAccelerationStructureBuildRangeInfoKHR* getRangeInfos(const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo);
	void encodeInputConversions(MVKCommandEncoder* cmdEncoder, MTLAccelerationStructureDescriptor* const* mtlDescs);
	void encodeInstanceConversion(MVKCommandEncoder* cmdEncoder,
								  id<MTLComputeCommandEncoder> mtlComputeEnc,
								  const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
								  MTLInstanceAccelerationStructureDescriptor* mtlInstDesc);
	void encodeTransformConversion(MVKCommandEncoder* cmdEncoder,
								   id<MTLComputeCommandEncoder> mtlComputeEnc,
								   const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
								   MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc);
	void encodeVertexAlignment(MVKCommandEncoder* cmdEncoder,
							   id<MTLComputeCommandEncoder> mtlComputeEnc,
							   const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
							   MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc);
	void encodeBoundingBoxConversion(MVKCommandEncoder* cmdEncoder,
									 id<MTLComputeCommandEncoder> mtlComputeEnc,
									 const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
									 MTLPrimitiveAccelerationStructureDescriptor* mtlPrimDesc);
	void encodeBuilds(MVKCommandEncoder* cmdEncoder, MTLAccelerationStructureDescriptor* const* mtlDescs);

	// The geometries and build ranges of all build infos, which reference their geometries in _geometries.
	MVKSmallVector<VkAccelerationStructureBuildGeometryInfoKHR, 1> _buildInfos;
	MVKSmallVector<VkAccelerationStructureGeometryKHR, 1> _geometries;
	MVKSmallVector<VkAccelerationStructureBuildRangeInfoKHR, 1> _rangeInfos;
};


#pragma mark -
#pragma mark MVKCmdCopyAccelerationStructure

/** Vulkan command to copy or compact an acceleration structure. */
class MVKCmdCopyAccelerationStructure : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						const VkCopyAccelerationStructureInfoKHR* pInfo);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	VkAccelerationStructureKHR _src;
	VkAccelerationStructureKHR _dst;
	VkCopyAccelerationStructureModeKHR _mode;
};


#pragma mark -
#pragma mark MVKCmdWriteAccelerationStructuresProperties

/** Vulkan command to write the properties of acceleration structures to queries. */
class MVKCmdWriteAccelerationStructuresProperties : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						uint32_t accelerationStructureCount,
						const VkAccelerationStructureKHR* pAccelerationStructures,
						VkQueryType queryType,
						VkQueryPool queryPool,
						uint32_t firstQuery);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	MVKSmallVector<VkAccelerationStructureKHR, 1> _accelerationStructures;
	VkQueryPool _queryPool;
	uint32_t _firstQuery;
};
