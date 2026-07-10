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

std::shared_mutex MVKAccelerationStructure::_mtlAccelerationStructureMapLock;
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
		std::lock_guard<std::shared_mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap[_mtlAccelerationStructure.gpuResourceID._impl] = this;
	}
}

void MVKAccelerationStructure::setMTLAccelerationStructure(id<MTLAccelerationStructure> mtlAS) {
	if (_mtlAccelerationStructure == mtlAS) { return; }

	if (_mtlAccelerationStructure) {
		std::lock_guard<std::shared_mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap.erase(_mtlAccelerationStructure.gpuResourceID._impl);
	}

	[_mtlAccelerationStructure release];
	_mtlAccelerationStructure = [mtlAS retain];

	if (_mtlAccelerationStructure) {
		std::lock_guard<std::shared_mutex> lock(_mtlAccelerationStructureMapLock);
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
	std::lock_guard<std::mutex> lock(_metadataLock);
	if (_instanceShaderBindingTableOffsetBuffer == mtlBuffer) { return; }
	[_instanceShaderBindingTableOffsetBuffer release];
	_instanceShaderBindingTableOffsetBuffer = [mtlBuffer retain];
}

id<MTLBuffer> MVKAccelerationStructure::getInstanceShaderBindingTableOffsetBuffer() {
	std::lock_guard<std::mutex> lock(_metadataLock);
	return _instanceShaderBindingTableOffsetBuffer;
}

MVKAccelerationStructure* MVKAccelerationStructure::getMVKAccelerationStructure(id<MTLAccelerationStructure> mtlAS) {
	if (!mtlAS) { return nullptr; }
	return getMVKAccelerationStructure(mtlAS.gpuResourceID._impl);
}

MVKAccelerationStructure* MVKAccelerationStructure::getMVKAccelerationStructure(VkDeviceAddress deviceAddress) {
	if (!deviceAddress) { return nullptr; }
	std::shared_lock<std::shared_mutex> lock(_mtlAccelerationStructureMapLock);
	auto it = _mtlAccelerationStructureMap.find(deviceAddress);
	return it == _mtlAccelerationStructureMap.end() ? nullptr : it->second;
}

void MVKAccelerationStructure::retainBuffer(id<MTLBuffer> mtlBuffer) {
	if (!mtlBuffer) { return; }
	std::lock_guard<std::mutex> lock(_metadataLock);
	[mtlBuffer retain];
	_retainedMTLBuffers.push_back(mtlBuffer);
}

void MVKAccelerationStructure::clearRetainedBuffers() {
	std::lock_guard<std::mutex> lock(_metadataLock);
	for (auto& mtlBuffer : _retainedMTLBuffers) {
		[mtlBuffer release];
	}
	_retainedMTLBuffers.clear();
	for (auto& blas : _referencedBLASes) {
		[blas release];
	}
	_referencedBLASes.clear();
	[_instanceShaderBindingTableOffsetBuffer release];
	_instanceShaderBindingTableOffsetBuffer = nil;
}

void MVKAccelerationStructure::setReferencedBLASes(NSArray<id<MTLAccelerationStructure>>* blasArray) {
	std::lock_guard<std::mutex> lock(_metadataLock);
	for (auto& blas : _referencedBLASes) {
		[blas release];
	}
	_referencedBLASes.clear();
	_referencedBLASes.reserve(blasArray.count);
	for (id<MTLAccelerationStructure> blas in blasArray) {
		[blas retain];
		_referencedBLASes.push_back(blas);
	}
}

void MVKAccelerationStructure::encodeResourceUsage(id<MTLComputeCommandEncoder> mtlEncoder) {
	std::lock_guard<std::mutex> lock(_metadataLock);
	if (_mtlAccelerationStructure) {
		[mtlEncoder useResource: _mtlAccelerationStructure usage: MTLResourceUsageRead];
	}
	for (auto& blas : _referencedBLASes) {
		[mtlEncoder useResource: blas usage: MTLResourceUsageRead];
	}
}

void MVKAccelerationStructure::copyRetainedBuffersFrom(MVKAccelerationStructure* srcAS) {
	if (srcAS == this) { return; }

	std::vector<id<MTLBuffer>> retainedBuffers;
	std::vector<id<MTLAccelerationStructure>> referencedBLASes;
	id<MTLBuffer> sbtOffsetBuffer = nil;
	if (srcAS) {
		std::lock_guard<std::mutex> srcLock(srcAS->_metadataLock);
		retainedBuffers.reserve(srcAS->_retainedMTLBuffers.size());
		for (auto& mtlBuffer : srcAS->_retainedMTLBuffers) {
			retainedBuffers.push_back([mtlBuffer retain]);
		}
		referencedBLASes.reserve(srcAS->_referencedBLASes.size());
		for (auto& blas : srcAS->_referencedBLASes) {
			referencedBLASes.push_back([blas retain]);
		}
		sbtOffsetBuffer = [srcAS->_instanceShaderBindingTableOffsetBuffer retain];
	}

	std::lock_guard<std::mutex> lock(_metadataLock);
	for (auto& mtlBuffer : _retainedMTLBuffers) {
		[mtlBuffer release];
	}
	for (auto& blas : _referencedBLASes) {
		[blas release];
	}
	[_instanceShaderBindingTableOffsetBuffer release];
	_retainedMTLBuffers = std::move(retainedBuffers);
	_referencedBLASes = std::move(referencedBLASes);
	_instanceShaderBindingTableOffsetBuffer = sbtOffsetBuffer;
}

MVKAccelerationStructure::~MVKAccelerationStructure() {
	clearRetainedBuffers();
	if (_mtlAccelerationStructure) {
		std::lock_guard<std::shared_mutex> lock(_mtlAccelerationStructureMapLock);
		_mtlAccelerationStructureMap.erase(_mtlAccelerationStructure.gpuResourceID._impl);
	}
	[_mtlAccelerationStructure release];
}
