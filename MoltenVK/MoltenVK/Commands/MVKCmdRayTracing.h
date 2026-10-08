/*
 * MVKCmdRayTracing.h
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

#pragma once

#include "MVKCommand.h"

#import <Metal/Metal.h>


/**
 * The parameters of a ray tracing dispatch, read by the ray tracing pipeline on the GPU.
 *
 * This struct must match spvRTDispatchParams of the SPIRV-Cross ray tracing pipeline header.
 */
struct MVKRayTracingDispatchParams {
	uint64_t raygenAddress;
	uint64_t raygenStride;
	uint64_t missAddress;
	uint64_t missStride;
	uint64_t hitAddress;
	uint64_t hitStride;
	uint64_t callableAddress;
	uint64_t callableStride;
	uint32_t launchWidth;
	uint32_t launchHeight;
	uint32_t launchDepth;
	uint32_t maxRecursionDepth;
	uint64_t indirectLaunchAddress;		/**< Address of the VkTraceRaysIndirectCommandKHR, or zero for a direct dispatch. */
};
static_assert(offsetof(MVKRayTracingDispatchParams, launchWidth) == 64, "MVKRayTracingDispatchParams must match spvRTDispatchParams.");
static_assert(offsetof(MVKRayTracingDispatchParams, indirectLaunchAddress) == 80, "MVKRayTracingDispatchParams must match spvRTDispatchParams.");
static_assert(sizeof(MVKRayTracingDispatchParams) == 88, "MVKRayTracingDispatchParams must match spvRTDispatchParams.");


#pragma mark -
#pragma mark MVKCmdTraceRays

/** Vulkan command to trace rays. */
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

	MVKRayTracingDispatchParams _params;
};


#pragma mark -
#pragma mark MVKCmdTraceRaysIndirect

/** Vulkan command to trace rays, with the launch size read from device memory. */
class MVKCmdTraceRaysIndirect : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
						const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
						const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
						const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
						VkDeviceAddress indirectDeviceAddress);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	MVKRayTracingDispatchParams _params;
	id<MTLBuffer> _mtlIndirectBuffer;
	VkDeviceSize _mtlIndirectBufferOffset;
};
