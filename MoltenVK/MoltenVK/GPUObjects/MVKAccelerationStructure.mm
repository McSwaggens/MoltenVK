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
#include "MVKCommandEncoderState.h"

using namespace std;


// The number of acceleration structure headers held by each MTLBuffer of a header pool.
static constexpr uint32_t kMVKAccelerationStructureHeadersPerMTLBuffer = 256;


#pragma mark -
#pragma mark MVKAccelerationStructure

id<MTLBuffer> MVKAccelerationStructure::getInstanceSBTOffsetsMTLBuffer(uint32_t instanceCount) {
	lock_guard<mutex> lock(_lock);

	NSUInteger length = max(instanceCount, 1u) * sizeof(uint32_t);
	if ( !_instanceSBTOffsetsMTLBuffers.empty() ) {
		NSUInteger currLength = _instanceSBTOffsetsMTLBuffers.back().length;
		if (currLength >= length) { return _instanceSBTOffsetsMTLBuffers.back(); }
		length = max(length, currLength * 2);	// Grow geometrically to limit the number of retained buffers
	}

	id<MTLBuffer> mtlBuff = [getMTLDevice() newBufferWithLength: length options: MTLResourceStorageModePrivate];	// retained
	if ( !mtlBuff ) { return nil; }

	setMetalObjectLabel(mtlBuff, @"Acceleration Structure Instance SBT Offsets");
	_device->makeResident(mtlBuff);
	_instanceSBTOffsetsMTLBuffers.push_back(mtlBuff);
	return mtlBuff;
}

id<MTLBuffer> MVKAccelerationStructure::getInstanceShaderBindingTableOffsetBuffer() {
	lock_guard<mutex> lock(_lock);
	return _instanceSBTOffsetsMTLBuffers.empty() ? nil : _instanceSBTOffsetsMTLBuffers.back();
}

void MVKAccelerationStructure::encodeResourceUsage(MVKUseResourceHelper& rez, MVKResourceUsageStages stage) {
	rez.add(_mtlAccelerationStructure, stage, false);

	lock_guard<mutex> lock(_lock);
	for (id<MTLBuffer> mtlBuff : _instanceSBTOffsetsMTLBuffers) {
		rez.add(mtlBuff, stage, false);
	}
}

void MVKAccelerationStructure::propagateDebugName() {
	setMetalObjectLabel(_mtlAccelerationStructure, _debugName);
}


#pragma mark Construction

MVKAccelerationStructure::MVKAccelerationStructure(MVKDevice* device,
												   const VkAccelerationStructureCreateInfoKHR* pCreateInfo) : MVKVulkanAPIDeviceObject(device) {

	// Metal allocates the memory of acceleration structures, so the buffer provided by the app is not used.
	_mtlAccelerationStructure = [getMTLDevice() newAccelerationStructureWithSize: pCreateInfo->size];	// retained
	if ( !_mtlAccelerationStructure ) {
		setConfigurationResult(reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY,
										   "vkCreateAccelerationStructureKHR(): Could not allocate a Metal acceleration structure of %llu bytes.",
										   pCreateInfo->size));
		return;
	}
	_device->makeResident(_mtlAccelerationStructure);
	setConfigurationResult(_device->getAccelerationStructureHeaderPool()->addAccelerationStructure(this));
}

MVKAccelerationStructure::~MVKAccelerationStructure() {
	if (_headerMTLBuffer) { _device->getAccelerationStructureHeaderPool()->removeAccelerationStructure(this); }
	for (id<MTLBuffer> mtlBuff : _instanceSBTOffsetsMTLBuffers) {
		_device->removeResidency(mtlBuff);
		[mtlBuff release];
	}
	if (_mtlAccelerationStructure) {
		_device->removeResidency(_mtlAccelerationStructure);
		[_mtlAccelerationStructure release];
	}
}


#pragma mark -
#pragma mark MVKAccelerationStructureHeaderPool

VkResult MVKAccelerationStructureHeaderPool::addAccelerationStructure(MVKAccelerationStructure* mvkAccStruct) {
	lock_guard<mutex> lock(_lock);

	if (_freeHeaderIndices.empty()) {
		id<MTLBuffer> mtlBuff = [getMTLDevice() newBufferWithLength: kMVKAccelerationStructureHeadersPerMTLBuffer * sizeof(MVKAccelerationStructureHeader)
															options: MTLResourceStorageModeShared];	// retained
		if ( !mtlBuff ) {
			return reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkCreateAccelerationStructureKHR(): Could not allocate acceleration structure headers.");
		}
		[mtlBuff setLabel: @"Acceleration Structure Headers"];
		_device->makeResident(mtlBuff);
		_mtlBuffers.push_back(mtlBuff);

		// Push in reverse, so lower header indices are allocated first.
		uint32_t endHdrIdx = (uint32_t)_accelerationStructures.size() + kMVKAccelerationStructureHeadersPerMTLBuffer;
		_accelerationStructures.resize(endHdrIdx, nullptr);
		for (uint32_t i = 0; i < kMVKAccelerationStructureHeadersPerMTLBuffer; i++) {
			_freeHeaderIndices.push_back(endHdrIdx - i - 1);
		}
	}

	uint32_t hdrIdx = _freeHeaderIndices.back();
	_freeHeaderIndices.pop_back();
	_accelerationStructures[hdrIdx] = mvkAccStruct;

	mvkAccStruct->_headerIndex = hdrIdx;
	mvkAccStruct->_headerMTLBuffer = _mtlBuffers[hdrIdx / kMVKAccelerationStructureHeadersPerMTLBuffer];
	mvkAccStruct->_headerOffset = (hdrIdx % kMVKAccelerationStructureHeadersPerMTLBuffer) * sizeof(MVKAccelerationStructureHeader);

	auto* pHeader = (MVKAccelerationStructureHeader*)((uintptr_t)mvkAccStruct->_headerMTLBuffer.contents + mvkAccStruct->_headerOffset);
	*pHeader = { mvkAccStruct->_mtlAccelerationStructure.gpuResourceID, 0 };

	return VK_SUCCESS;
}

void MVKAccelerationStructureHeaderPool::removeAccelerationStructure(MVKAccelerationStructure* mvkAccStruct) {
	lock_guard<mutex> lock(_lock);

	auto* pHeader = (MVKAccelerationStructureHeader*)((uintptr_t)mvkAccStruct->_headerMTLBuffer.contents + mvkAccStruct->_headerOffset);
	*pHeader = {};

	_accelerationStructures[mvkAccStruct->_headerIndex] = nullptr;
	_freeHeaderIndices.push_back(mvkAccStruct->_headerIndex);
}

void MVKAccelerationStructureHeaderPool::encodeResourceUsage(MVKUseResourceHelper& rez, MVKResourceUsageStages stage) {
	lock_guard<mutex> lock(_lock);

	for (id<MTLBuffer> mtlBuff : _mtlBuffers) {
		rez.add(mtlBuff, stage, false);
	}
	for (MVKAccelerationStructure* mvkAccStruct : _accelerationStructures) {
		if (mvkAccStruct) { mvkAccStruct->encodeResourceUsage(rez, stage); }
	}
}

MVKAccelerationStructureHeaderPool::~MVKAccelerationStructureHeaderPool() {
	for (id<MTLBuffer> mtlBuff : _mtlBuffers) {
		_device->removeResidency(mtlBuff);
		[mtlBuff release];
	}
}
