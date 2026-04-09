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

std::mutex MVKAccelerationStructure::_mtlAccelerationStructureMapLock;
std::unordered_map<uint64_t, MVKAccelerationStructure*> MVKAccelerationStructure::_mtlAccelerationStructureMap;

MVKAccelerationStructure::MVKAccelerationStructure(MVKDevice* device,
                                                   const VkAccelerationStructureCreateInfoKHR* pCreateInfo)
	: MVKVulkanAPIDeviceObject(device) {

	_type = pCreateInfo->type;
	_buffer = pCreateInfo->buffer;
	_offset = pCreateInfo->offset;
	_size = pCreateInfo->size;

	// Allocate the Metal acceleration structure with the requested size.
	_mtlAccelerationStructure = [getMTLDevice() newAccelerationStructureWithSize: (NSUInteger)_size];
	if (_mtlAccelerationStructure) {
		std::lock_guard<std::mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap[_mtlAccelerationStructure.gpuResourceID._impl] = this;
	}
}

void MVKAccelerationStructure::setMTLAccelerationStructure(id<MTLAccelerationStructure> mtlAS) {
	if (_mtlAccelerationStructure == mtlAS) { return; }

	if (_mtlAccelerationStructure) {
		std::lock_guard<std::mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap.erase(_mtlAccelerationStructure.gpuResourceID._impl);
	}

	[_mtlAccelerationStructure release];
	_mtlAccelerationStructure = [mtlAS retain];

	if (_mtlAccelerationStructure) {
		std::lock_guard<std::mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap[_mtlAccelerationStructure.gpuResourceID._impl] = this;
	}
}

VkDeviceAddress MVKAccelerationStructure::getDeviceAddress() {
	if (_mtlAccelerationStructure) {
		return _mtlAccelerationStructure.gpuResourceID._impl;
	}
	return 0;
}

void MVKAccelerationStructure::setInstanceShaderBindingTableOffsetBuffer(id<MTLBuffer> mtlBuffer) {
	if (_instanceShaderBindingTableOffsetBuffer == mtlBuffer) { return; }
	[_instanceShaderBindingTableOffsetBuffer release];
	_instanceShaderBindingTableOffsetBuffer = [mtlBuffer retain];
}

MVKAccelerationStructure* MVKAccelerationStructure::getMVKAccelerationStructure(id<MTLAccelerationStructure> mtlAS) {
	if (!mtlAS) { return nullptr; }
	std::lock_guard<std::mutex> lock(_mtlAccelerationStructureMapLock);
	auto it = _mtlAccelerationStructureMap.find(mtlAS.gpuResourceID._impl);
	return it == _mtlAccelerationStructureMap.end() ? nullptr : it->second;
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
	setInstanceShaderBindingTableOffsetBuffer(nil);
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
	if (_mtlAccelerationStructure) {
		std::lock_guard<std::mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap.erase(_mtlAccelerationStructure.gpuResourceID._impl);
	}
	[_mtlAccelerationStructure release];
}
