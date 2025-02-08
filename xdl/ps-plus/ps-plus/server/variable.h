/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PS_PLUS_SERVER_VARIABLE_H_
#define PS_PLUS_SERVER_VARIABLE_H_

#include "ps-plus/common/data.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/status.h"
#include "ps-plus/common/qrw_lock.h"
#include <memory>
#include <unordered_map>

namespace ps {
namespace server {

class Variable {
  public:
   enum SlotJoiner {
      kVariableLike,
      kAnyOne,
   };
   struct Slot {
      std::unique_ptr<Tensor> tensor;
      SlotJoiner joiner;
   };

   Variable(Tensor* data, Data* slicer, std::string name)
       : data_(data), slicer_(slicer), name_(name), real_inited_(false) {
   }

   // you should lock this when you process the data.
   QRWLock& VariableLock() { return variable_lock_; }

   // you should use following method when VariableLock is read_locked.
   Data* GetSlicer() { return slicer_.get(); }
   Tensor* GetData() {
      Tensor* tensor = data_.get();
      return tensor;
   }
   Tensor* GetSlot(const std::string& name, const std::function<Slot()>& slot_creator);
   Slot VariableLikeSlot(DataType type, const TensorShape& shape, Initializer* initializer);
   Slot AnyOneSlot(DataType type, const TensorShape& shape, Initializer* initializer);
   Tensor* GetVariableLikeSlot(const std::string& name, DataType type, const std::function<Initializer*()>& initializer_creator);
   Tensor* GetVariableLikeSlot(const std::string& name, DataType type, const TensorShape& inner_shape, const std::function<Initializer*()>& initializer_creator);
   Tensor* GetAnyOneSlot(const std::string& name, DataType type, const TensorShape& shape, const std::function<Initializer*()>& initializer_creator);
   Status GetExistSlot(const std::string& name, Tensor** result);
   Status ReShapeId(size_t id);
   void ClearIds(const std::vector<size_t>& id);
   std::string GetName() { return name_; }
   // Used for Save and Restore
   const std::unordered_map<std::string, Slot>& GetSlots() { return slots_; }
   void SetSlots(std::unordered_map<std::string, Slot>&& slots) { slots_ = std::move(slots); }
   bool RealInited() { return real_inited_; }
   void SetRealInited(bool init) { real_inited_ = init; }
   std::unique_ptr<Variable> Clone() const {
      auto cloned_variable = std::make_unique<Variable>(nullptr, nullptr, name_);
      cloned_variable->real_inited_ = real_inited_;
      cloned_variable->data_ = std::make_unique<Tensor>(data_->Clone());
      cloned_variable->slicer_ = std::make_unique<Data>(*slicer_.get());
      // 在 Clone 方法中，打印原始对象和克隆对象的内存地址
      // printf("Original data_ address: %p\n", data_.get());
      // printf("Cloned data_ address: %p\n", cloned_variable->data_.get());

      // printf("Original slicer_ address: %p\n", slicer_.get());
      // printf("Cloned slicer_ address: %p\n", cloned_variable->slicer_.get());

      // Deep copy the slots unordered_map
      // for (const auto& pair : slots_) {
      //   // Create a deep copy of each Slot
      //   Slot cloned_slot;
      //   cloned_slot.tensor = pair.second.tensor ? std::make_unique<Tensor>(*pair.second.tensor) : nullptr;
      //   cloned_slot.joiner = pair.second.joiner;

      //   // Insert the deep-copied Slot into the cloned variable's slots map
      //   cloned_variable->slots_[pair.first] = std::move(cloned_slot);
      // }

      return cloned_variable;
   }

  private:
   // There is 3 state in Variable Processor:
   // <variable_lock_.read, slots_lock_.read>
   // <variable_lock_.read, slots_lock_.write>
   // <variable_lock_.write, None>
   QRWLock variable_lock_;  // Guard variable
   QRWLock slots_lock_;     // Guard the slots unordered_map

   std::unique_ptr<Tensor> data_;
   std::unique_ptr<Data> slicer_;
   std::unordered_map<std::string, Slot> slots_;
   std::string name_;
   bool real_inited_;
};

}  // namespace server
}  // namespace ps

#endif
