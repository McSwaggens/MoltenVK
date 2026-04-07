/*
 * MVKAccelerationStructure.mm
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

#include "MVKAccelerationStructure.h"
#include "MVKBuffer.h"

MVKAccelerationStructure::MVKAccelerationStructure(MVKDevice* device,
                                                   const VkAccelerationStructureCreateInfoKHR* pCreateInfo)
	: MVKVulkanAPIDeviceObject(device) {

	_type = pCreateInfo->type;
	_buffer = pCreateInfo->buffer;
	_offset = pCreateInfo->offset;
	_size = pCreateInfo->size;

	// Allocate the Metal acceleration structure with the requested size.
	_mtlAccelerationStructure = [getMTLDevice() newAccelerationStructureWithSize: (NSUInteger)_size];
}

VkDeviceAddress MVKAccelerationStructure::getDeviceAddress() {
	if (_mtlAccelerationStructure) {
		return _mtlAccelerationStructure.gpuResourceID._impl;
	}
	return 0;
}

void MVKAccelerationStructure::retainBuffer(id<MTLBuffer> mtlBuffer) {
	if (!mtlBuffer) { return; }
	[mtlBuffer retain];
	_retainedMTLBuffers.push_back(mtlBuffer);
}

void MVKAccelerationStructure::clearRetainedBuffers() {
	for (auto& mtlBuffer : _retainedMTLBuffers) {
		[mtlBuffer release];
	}
	_retainedMTLBuffers.clear();
}

void MVKAccelerationStructure::copyRetainedBuffersFrom(MVKAccelerationStructure* srcAS) {
	clearRetainedBuffers();
	if (!srcAS) { return; }
	for (auto& mtlBuffer : srcAS->_retainedMTLBuffers) {
		retainBuffer(mtlBuffer);
	}
}

MVKAccelerationStructure::~MVKAccelerationStructure() {
	clearRetainedBuffers();
	[_mtlAccelerationStructure release];
}
