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

class MVKCmdBuildAccelerationStructures : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						uint32_t infoCount,
						const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
						const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	struct BuildInfo {
		VkAccelerationStructureBuildGeometryInfoKHR geometryInfo;
		MVKSmallVector<VkAccelerationStructureGeometryKHR> geometries;
		MVKSmallVector<VkAccelerationStructureBuildRangeInfoKHR> buildRangeInfos;
	};

	MVKSmallVector<BuildInfo> _buildInfos;
};


#pragma mark -
#pragma mark MVKCmdCopyAccelerationStructure

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

	MVKSmallVector<VkAccelerationStructureKHR> _accelerationStructures;
	VkQueryType _queryType;
	VkQueryPool _queryPool;
	uint32_t _firstQuery;
};
