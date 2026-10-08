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

/**
 * The memory holding the header of an acceleration structure. Vulkan requires the device addresses of acceleration
 * structures to be aligned to 256 bytes, so each header has a slot of that size, which also holds data only used
 * by MoltenVK's acceleration structure commands.
 */
typedef struct {
	MVKAccelerationStructureHeader header;
	uint32_t instanceCount;		/**< For a TLAS, the number of SBT record offsets at header.instanceSBTOffsets. */
} MVKAccelerationStructureHeaderSlot;

/** The size and alignment of each acceleration structure header slot. */
static constexpr NSUInteger kMVKAccelerationStructureHeaderSlotSize = 256;

static_assert(sizeof(MVKAccelerationStructureHeaderSlot) <= kMVKAccelerationStructureHeaderSlotSize, "MVKAccelerationStructureHeaderSlot is too large.");


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

	/** Returns the type this acceleration structure was created with. */
	VkAccelerationStructureTypeKHR getType() { return _type; }

	/** Returns the MTLBuffer holding the header of this acceleration structure. */
	id<MTLBuffer> getHeaderMTLBuffer() { return _headerMTLBuffer; }

	/** Returns the offset of the header slot of this acceleration structure within its MTLBuffer. */
	NSUInteger getHeaderOffset() { return _headerOffset; }

	/**
	 * Returns the device address of this acceleration structure, which is the GPU address of its header.
	 *
	 * Because Metal allocates the memory of acceleration structures, and the header is allocated by MoltenVK,
	 * this address is unrelated to the buffer and offset the acceleration structure was created with. In particular,
	 * the relative offsets of acceleration structures of type VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR created in
	 * the same VkBuffer are not reflected in the differences of their device addresses, as Vulkan requires.
	 */
	uint64_t getDeviceAddress() { return _deviceAddress; }

	/**
	 * Returns a buffer that can hold the shader binding table record offsets of the specified number of
	 * instances, to be written by a build of, or a copy to, this top-level acceleration structure.
	 *
	 * The GPU stores the address of the buffer in the header, after writing the buffer. A buffer that is too
	 * small is replaced by a larger one, but is retained until this acceleration structure is destroyed,
	 * because previously encoded GPU work may still access it through the header.
	 */
	id<MTLBuffer> getInstanceSBTOffsetsMTLBuffer(uint32_t instanceCount);

	/**
	 * Returns the number of instance shader binding table record offsets that the largest buffer returned by
	 * getInstanceSBTOffsetsMTLBuffer() so far can hold, or zero if there is no such buffer.
	 */
	uint32_t getInstanceSBTOffsetsCapacity();

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
	uint64_t _deviceAddress = 0;
	uint32_t _headerIndex = 0;
	VkAccelerationStructureTypeKHR _type;
	MVKSmallVector<id<MTLBuffer>, 1> _instanceSBTOffsetsMTLBuffers;		// Most recent last. Guarded by the header pool lock.
};


#pragma mark -
#pragma mark MVKAccelerationStructureHeaderPool

/**
 * Tracks the live acceleration structures of a device, and allocates their headers
 * from MTLBuffers that are owned by this pool and remain resident for its lifetime.
 *
 * Because top-level acceleration structures reference bottom-level acceleration structures by device address,
 * any live acceleration structure may be accessed by the GPU, and all of them, with their header buffers and
 * instance data buffers, must be resident wherever acceleration structures are used. Without a residency set,
 * this pool makes them resident in Metal encoders while holding its lock, so another thread can never destroy
 * a Metal resource that is being used. Header buffers live as long as this pool, so they can be used later.
 */
class MVKAccelerationStructureHeaderPool : public MVKBaseDeviceObject {

public:

	/** Returns the Vulkan API opaque object controlling this object. */
	MVKVulkanAPIObject* getVulkanAPIObject() override { return _device; };

	/** Allocates a header for the acceleration structure, and writes its Metal acceleration structure to it. */
	VkResult addAccelerationStructure(MVKAccelerationStructure* mvkAccStruct);

	/** Clears and frees the header of the acceleration structure, after which its Metal resources are no longer used. */
	void removeAccelerationStructure(MVKAccelerationStructure* mvkAccStruct);

	/** Implements MVKAccelerationStructure::getInstanceSBTOffsetsMTLBuffer(). */
	id<MTLBuffer> getInstanceSBTOffsetsMTLBuffer(MVKAccelerationStructure* mvkAccStruct, uint32_t instanceCount);

	/** Implements MVKAccelerationStructure::getInstanceSBTOffsetsCapacity(). */
	uint32_t getInstanceSBTOffsetsCapacity(MVKAccelerationStructure* mvkAccStruct);

	/** Makes the header buffers and the Metal resources of all live acceleration structures resident in the compute encoder. */
	void useResources(id<MTLComputeCommandEncoder> mtlComputeEnc);

	/**
	 * Makes the Metal resources of all live acceleration structures resident in the render or compute encoder, by
	 * calling the function for each of them, and adds the header buffers to the resource helper to be used later.
	 */
	void useResources(id<MTLCommandEncoder> mtlEncoder, MVKUseMTLResourceFunction useResource,
					  MVKUseResourceHelper& rez, MVKResourceUsageStages stages);

	/** Makes the Metal acceleration structures of all live acceleration structures resident in the encoder. */
	void useMTLAccelerationStructures(id<MTLAccelerationStructureCommandEncoder> mtlASEnc);

	MVKAccelerationStructureHeaderPool(MVKDevice* device) : MVKBaseDeviceObject(device) {}

	~MVKAccelerationStructureHeaderPool() override;

protected:
	void updateMTLResources();

	MVKSmallVector<id<MTLBuffer>> _mtlBuffers;
	MVKSmallVector<NSUInteger> _mtlBufferSlotOffsets;			// Offset of the first 256-byte aligned header slot of each MTLBuffer
	MVKSmallVector<MVKAccelerationStructure*> _accelerationStructures;	// Indexed by header index, null if free
	MVKSmallVector<uint32_t> _freeHeaderIndices;
	MVKSmallVector<id<MTLResource>> _mtlResources;	// The Metal acceleration structures, then the instance data buffers, of all live acceleration structures
	size_t _mtlAccelerationStructureCount = 0;		// The number of Metal acceleration structures at the start of _mtlResources
	uint64_t _generation = 1;						// Changes whenever the content of _mtlResources would change
	uint64_t _mtlResourcesGeneration = 0;			// The generation _mtlResources was gathered at
	std::mutex _lock;
};
