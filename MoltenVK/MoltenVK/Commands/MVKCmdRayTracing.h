/*
 * MVKCmdRayTracing.h
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


#pragma mark -
#pragma mark MVKCmdTraceRays

class MVKCmdTraceRays : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
						const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
						const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
						const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
						uint32_t width,
						uint32_t height,
						uint32_t depth);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	VkStridedDeviceAddressRegionKHR _raygenSBT;
	VkStridedDeviceAddressRegionKHR _missSBT;
	VkStridedDeviceAddressRegionKHR _hitSBT;
	VkStridedDeviceAddressRegionKHR _callableSBT;
	uint32_t _width;
	uint32_t _height;
	uint32_t _depth;
};


#pragma mark -
#pragma mark MVKCmdBindRayTracingPipeline

class MVKCmdBindRayTracingPipeline : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff, VkPipeline pipeline);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	VkPipeline _pipeline;
};
