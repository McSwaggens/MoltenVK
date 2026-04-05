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

	// Make acceleration structures resident. Skip if none exist.
	auto& allAS = cmdEncoder->getDevice()->getAccelerationStructures();
	if (!allAS.empty()) {
		for (auto* as : allAS) {
			if (as && as->getMTLAccelerationStructure()) {
				[mtlEncoder useResource: as->getMTLAccelerationStructure() usage: MTLResourceUsageRead];
			}
		}
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
