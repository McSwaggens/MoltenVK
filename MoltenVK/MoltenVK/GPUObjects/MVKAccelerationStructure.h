/*
 * MVKAccelerationStructure.h
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

#include "MVKDevice.h"
#include <vector>

#import <Metal/Metal.h>


#pragma mark -
#pragma mark MVKAccelerationStructure

class MVKAccelerationStructure : public MVKVulkanAPIDeviceObject {

public:

	VkObjectType getVkObjectType() override { return VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR; }

	VkDebugReportObjectTypeEXT getVkDebugReportObjectType() override { return VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT; }

	/** Returns the Metal acceleration structure. May be nil if not yet built. */
	id<MTLAccelerationStructure> getMTLAccelerationStructure() { return _mtlAccelerationStructure; }

	/** Sets the Metal acceleration structure (called during build). */
	void setMTLAccelerationStructure(id<MTLAccelerationStructure> mtlAS) { _mtlAccelerationStructure = mtlAS; }

	/** Retains a Metal buffer that must remain alive for this acceleration structure. */
	void retainBuffer(id<MTLBuffer> mtlBuffer);

	/** Releases all retained Metal buffers owned by this acceleration structure. */
	void clearRetainedBuffers();

	/** Shares retained Metal buffers from another acceleration structure copy source. */
	void copyRetainedBuffersFrom(MVKAccelerationStructure* srcAS);

	/** Returns the device address for this acceleration structure. */
	VkDeviceAddress getDeviceAddress();

	/** Returns the acceleration structure type (top-level or bottom-level). */
	VkAccelerationStructureTypeKHR getType() { return _type; }

	/** Returns the size of this acceleration structure. */
	VkDeviceSize getSize() { return _size; }

	MVKAccelerationStructure(MVKDevice* device, const VkAccelerationStructureCreateInfoKHR* pCreateInfo);

	~MVKAccelerationStructure() override;

protected:
	void propagateDebugName() override {}

	id<MTLAccelerationStructure> _mtlAccelerationStructure = nil;
	VkAccelerationStructureTypeKHR _type;
	VkBuffer _buffer;
	VkDeviceSize _offset;
	VkDeviceSize _size;
	std::vector<id<MTLBuffer>> _retainedMTLBuffers;
};
