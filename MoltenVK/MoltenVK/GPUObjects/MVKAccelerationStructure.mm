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


// The number of acceleration structure header slots held by each MTLBuffer of a header pool.
static constexpr uint32_t kMVKAccelerationStructureHeadersPerMTLBuffer = 256;


#pragma mark -
#pragma mark MVKAccelerationStructure

id<MTLBuffer> MVKAccelerationStructure::getInstanceSBTOffsetsMTLBuffer(uint32_t instanceCount) {
	return _device->getAccelerationStructureHeaderPool()->getInstanceSBTOffsetsMTLBuffer(this, instanceCount);
}

uint32_t MVKAccelerationStructure::getInstanceSBTOffsetsCapacity() {
	return _device->getAccelerationStructureHeaderPool()->getInstanceSBTOffsetsCapacity(this);
}

void MVKAccelerationStructure::propagateDebugName() {
	setMetalObjectLabel(_mtlAccelerationStructure, _debugName);
}


#pragma mark Construction

MVKAccelerationStructure::MVKAccelerationStructure(MVKDevice* device,
												   const VkAccelerationStructureCreateInfoKHR* pCreateInfo) : MVKVulkanAPIDeviceObject(device) {
	_type = pCreateInfo->type;

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

// Once removed from the header pool, the Metal resources are no longer used by encoders, and can be released.
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
		// Leave room to align the first header slot, in case the MTLBuffer is not aligned to the slot size.
		NSUInteger mtlBuffLen = (kMVKAccelerationStructureHeadersPerMTLBuffer + 1) * kMVKAccelerationStructureHeaderSlotSize;
		id<MTLBuffer> mtlBuff = [getMTLDevice() newBufferWithLength: mtlBuffLen options: MTLResourceStorageModeShared];	// retained
		if ( !mtlBuff ) {
			return reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkCreateAccelerationStructureKHR(): Could not allocate acceleration structure headers.");
		}
		[mtlBuff setLabel: @"Acceleration Structure Headers"];
		_device->makeResident(mtlBuff);
		_device->getLiveResources().add(mtlBuff);	// Descriptors reference headers like buffers
		_mtlBuffers.push_back(mtlBuff);
		_mtlBufferSlotOffsets.push_back(mvkAlignByteCount(mtlBuff.gpuAddress, kMVKAccelerationStructureHeaderSlotSize) - mtlBuff.gpuAddress);

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
	_generation++;

	uint32_t mtlBuffIdx = hdrIdx / kMVKAccelerationStructureHeadersPerMTLBuffer;
	mvkAccStruct->_headerIndex = hdrIdx;
	mvkAccStruct->_headerMTLBuffer = _mtlBuffers[mtlBuffIdx];
	mvkAccStruct->_headerOffset = (_mtlBufferSlotOffsets[mtlBuffIdx] +
								   (hdrIdx % kMVKAccelerationStructureHeadersPerMTLBuffer) * kMVKAccelerationStructureHeaderSlotSize);
	mvkAccStruct->_deviceAddress = mvkAccStruct->_headerMTLBuffer.gpuAddress + mvkAccStruct->_headerOffset;

	auto* pSlot = (MVKAccelerationStructureHeaderSlot*)((uintptr_t)mvkAccStruct->_headerMTLBuffer.contents + mvkAccStruct->_headerOffset);
	*pSlot = {};
	pSlot->header.accelerationStructure = mvkAccStruct->_mtlAccelerationStructure.gpuResourceID;

	return VK_SUCCESS;
}

void MVKAccelerationStructureHeaderPool::removeAccelerationStructure(MVKAccelerationStructure* mvkAccStruct) {
	lock_guard<mutex> lock(_lock);

	auto* pSlot = (MVKAccelerationStructureHeaderSlot*)((uintptr_t)mvkAccStruct->_headerMTLBuffer.contents + mvkAccStruct->_headerOffset);
	*pSlot = {};

	_accelerationStructures[mvkAccStruct->_headerIndex] = nullptr;
	_freeHeaderIndices.push_back(mvkAccStruct->_headerIndex);
	_generation++;
}

id<MTLBuffer> MVKAccelerationStructureHeaderPool::getInstanceSBTOffsetsMTLBuffer(MVKAccelerationStructure* mvkAccStruct, uint32_t instanceCount) {
	lock_guard<mutex> lock(_lock);

	auto& mtlBuffs = mvkAccStruct->_instanceSBTOffsetsMTLBuffers;
	NSUInteger length = max(instanceCount, 1u) * sizeof(uint32_t);
	if ( !mtlBuffs.empty() ) {
		NSUInteger currLength = mtlBuffs.back().length;
		if (currLength >= length) { return mtlBuffs.back(); }
		length = max(length, currLength * 2);	// Grow geometrically to limit the number of retained buffers
	}

	id<MTLBuffer> mtlBuff = [getMTLDevice() newBufferWithLength: length options: MTLResourceStorageModePrivate];	// retained
	if ( !mtlBuff ) { return nil; }

	[mtlBuff setLabel: @"Acceleration Structure Instance SBT Offsets"];
	_device->makeResident(mtlBuff);
	mtlBuffs.push_back(mtlBuff);
	_generation++;
	return mtlBuff;
}

uint32_t MVKAccelerationStructureHeaderPool::getInstanceSBTOffsetsCapacity(MVKAccelerationStructure* mvkAccStruct) {
	lock_guard<mutex> lock(_lock);

	auto& mtlBuffs = mvkAccStruct->_instanceSBTOffsetsMTLBuffers;
	return mtlBuffs.empty() ? 0 : uint32_t(mtlBuffs.back().length / sizeof(uint32_t));
}

// Gathers the Metal resources of all live acceleration structures, if they have changed since they were last gathered.
// Must be called while holding the lock.
void MVKAccelerationStructureHeaderPool::updateMTLResources() {
	if (_mtlResourcesGeneration == _generation) { return; }

	_mtlResources.clear();
	for (MVKAccelerationStructure* mvkAccStruct : _accelerationStructures) {
		if (mvkAccStruct) { _mtlResources.push_back(mvkAccStruct->_mtlAccelerationStructure); }
	}
	_mtlAccelerationStructureCount = _mtlResources.size();
	for (MVKAccelerationStructure* mvkAccStruct : _accelerationStructures) {
		if (mvkAccStruct) {
			for (id<MTLBuffer> mtlBuff : mvkAccStruct->_instanceSBTOffsetsMTLBuffers) { _mtlResources.push_back(mtlBuff); }
		}
	}
	_mtlResourcesGeneration = _generation;
}

void MVKAccelerationStructureHeaderPool::useResources(id<MTLComputeCommandEncoder> mtlComputeEnc) {
	lock_guard<mutex> lock(_lock);

	updateMTLResources();
	if ( !_mtlBuffers.empty() ) {
		[mtlComputeEnc useResources: (const id<MTLResource>*)_mtlBuffers.data() count: _mtlBuffers.size() usage: MTLResourceUsageRead];
	}
	if ( !_mtlResources.empty() ) {
		[mtlComputeEnc useResources: _mtlResources.data() count: _mtlResources.size() usage: MTLResourceUsageRead];
	}
}

void MVKAccelerationStructureHeaderPool::useResources(id<MTLCommandEncoder> mtlEncoder, MVKUseMTLResourceFunction useResource,
													  MVKUseResourceHelper& rez, MVKResourceUsageStages stages) {
	lock_guard<mutex> lock(_lock);

	updateMTLResources();
	for (id<MTLResource> mtlRez : _mtlResources) {
		useResource(mtlEncoder, mtlRez, MTLResourceUsageRead, stages);
	}
	for (id<MTLBuffer> mtlBuff : _mtlBuffers) {
		rez.add(mtlBuff, stages, false);
	}
}

void MVKAccelerationStructureHeaderPool::useMTLAccelerationStructures(id<MTLAccelerationStructureCommandEncoder> mtlASEnc) {
	lock_guard<mutex> lock(_lock);

	updateMTLResources();
	if (_mtlAccelerationStructureCount) {
		[mtlASEnc useResources: _mtlResources.data() count: _mtlAccelerationStructureCount usage: MTLResourceUsageRead];
	}
}

MVKAccelerationStructureHeaderPool::~MVKAccelerationStructureHeaderPool() {
	for (id<MTLBuffer> mtlBuff : _mtlBuffers) {
		_device->getLiveResources().remove(mtlBuff);
		_device->removeResidency(mtlBuff);
		[mtlBuff release];
	}
}
