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
#include "MVKSmallVector.h"
#include <mutex>

#import <Metal/Metal.h>


/**
 * The device-memory representation of a Vulkan acceleration structure, whose GPU address is the
 * VkDeviceAddress of the acceleration structure. Shaders see it as struct spvAccelerationStructure.
 */
typedef struct {
	MTLResourceID accelerationStructure;	/**< The Metal acceleration structure. */
	uint64_t instanceSBTOffsets;			/**< For a TLAS, the GPU address of the SBT record offsets of its instances. */
} MVKAccelerationStructureHeader;

static_assert(sizeof(MVKAccelerationStructureHeader) == 16, "MVKAccelerationStructureHeader must match spvAccelerationStructure.");


#pragma mark -
#pragma mark MVKAccelerationStructure

/** Represents a Vulkan acceleration structure. */
class MVKAccelerationStructure : public MVKVulkanAPIDeviceObject {

public:

	/** Returns the Vulkan type of this object. */
	VkObjectType getVkObjectType() override { return VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR; }

	/** Returns the debug report object type of this object. */
	VkDebugReportObjectTypeEXT getVkDebugReportObjectType() override { return VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT; }

	/** Returns the Metal acceleration structure. */
	id<MTLAccelerationStructure> getMTLAccelerationStructure() { return _mtlAccelerationStructure; }

	/** Returns the MTLBuffer holding the header of this acceleration structure. */
	id<MTLBuffer> getHeaderMTLBuffer() { return _headerMTLBuffer; }

	/** Returns the offset of the header of this acceleration structure within its MTLBuffer. */
	NSUInteger getHeaderOffset() { return _headerOffset; }

	/** Returns the device address of this acceleration structure, which is the GPU address of its header. */
	uint64_t getDeviceAddress() { return _headerMTLBuffer.gpuAddress + _headerOffset; }

	/**
	 * Returns a buffer that can hold the shader binding table record offsets of the specified number of
	 * instances, to be written by a build of, or a copy to, this top-level acceleration structure.
	 *
	 * The GPU stores the address of the buffer in the header, after writing the buffer. A buffer that is too
	 * small is replaced by a larger one, but is retained until this acceleration structure is destroyed,
	 * because previously encoded GPU work may still access it through the header.
	 */
	id<MTLBuffer> getInstanceSBTOffsetsMTLBuffer(uint32_t instanceCount);

	/** Returns the buffer holding the instance SBT record offsets of the most recently encoded build or copy, or nil. */
	id<MTLBuffer> getInstanceShaderBindingTableOffsetBuffer();

	/** Adds the Metal acceleration structure and the instance SBT record offset buffers to the resource usage helper. */
	void encodeResourceUsage(MVKUseResourceHelper& rez, MVKResourceUsageStages stage);

	/** Returns the specified geometry of the build info, which may be held in either an array or an array of pointers. */
	static const VkAccelerationStructureGeometryKHR& getGeometry(const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
																 uint32_t geometryIndex) {
		return buildInfo.pGeometries ? buildInfo.pGeometries[geometryIndex] : *buildInfo.ppGeometries[geometryIndex];
	}

#pragma mark Construction

	MVKAccelerationStructure(MVKDevice* device, const VkAccelerationStructureCreateInfoKHR* pCreateInfo);

	~MVKAccelerationStructure() override;

protected:
	friend class MVKAccelerationStructureHeaderPool;

	void propagateDebugName() override;

	id<MTLAccelerationStructure> _mtlAccelerationStructure = nil;
	id<MTLBuffer> _headerMTLBuffer = nil;
	NSUInteger _headerOffset = 0;
	uint32_t _headerIndex = 0;
	MVKSmallVector<id<MTLBuffer>, 1> _instanceSBTOffsetsMTLBuffers;		// Most recent last
	std::mutex _lock;
};


#pragma mark -
#pragma mark MVKAccelerationStructureHeaderPool

/**
 * Tracks the live acceleration structures of a device, and allocates their headers
 * from MTLBuffers that are owned by this pool and remain resident for its lifetime.
 */
class MVKAccelerationStructureHeaderPool : public MVKBaseDeviceObject {

public:

	/** Returns the Vulkan API opaque object controlling this object. */
	MVKVulkanAPIObject* getVulkanAPIObject() override { return _device; };

	/** Allocates a header for the acceleration structure, and writes its Metal acceleration structure to it. */
	VkResult addAccelerationStructure(MVKAccelerationStructure* mvkAccStruct);

	/** Clears and frees the header of the acceleration structure. */
	void removeAccelerationStructure(MVKAccelerationStructure* mvkAccStruct);

	/** Adds the header buffers and the resources of all live acceleration structures to the resource usage helper. */
	void encodeResourceUsage(MVKUseResourceHelper& rez, MVKResourceUsageStages stage);

	/** Returns the live acceleration structure using the Metal acceleration structure, or null if there is none. */
	MVKAccelerationStructure* getAccelerationStructure(id<MTLAccelerationStructure> mtlAccStruct);

	MVKAccelerationStructureHeaderPool(MVKDevice* device) : MVKBaseDeviceObject(device) {}

	~MVKAccelerationStructureHeaderPool() override;

protected:
	MVKSmallVector<id<MTLBuffer>> _mtlBuffers;
	MVKSmallVector<MVKAccelerationStructure*> _accelerationStructures;	// Indexed by header index, null if free
	MVKSmallVector<uint32_t> _freeHeaderIndices;
	std::mutex _lock;
};
