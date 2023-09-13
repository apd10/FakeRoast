import torch
import numpy as np
import pdb
import mmh3 

''' hasher classes that can be used in actual kernels '''
class Hasher:
    def __init__(self, seed):
      self.seed = seed

    def hash(self, numbers):
      raise NotImplementedError


class MurmurHasher(Hasher):
    def __init__(self, seed, **kwargs):
        super(MurmurHasher, self).__init__(seed)
        self.seed = seed

    def hash1(self, numbers, target_size=None):
        device = numbers.device
        cpu_numbers = np.array(numbers.to("cpu"))
        hashed_numbers = np.array([ mmh3.hash(i, seed=self.seed) for i in cpu_numbers])
        hashed_numbers = torch.LongTensor(hashed_numbers).to(device) % target_size
        return hashed_numbers

    def hash2(self, number1, number2, target_size=None):
        device = number1.device
        output = np.zeros(shape=(number1.numel(), number2.numel()))
        cpu_num1 = np.array(number1.to("cpu")).reshape(-1)
        cpu_num2 = np.array(number2.to("cpu")).reshape(-1)
        for num1 in range(len(cpu_num1)):
            for num2 in range(len(cpu_num2)):
                output[num1,num2] = mmh3.hash(str(cpu_num1[num1])+"-"+str(cpu_num2[num2]), seed=self.seed, signed=False)
        return torch.LongTensor(output).to(device) %target_size

class UHasher(Hasher):
    def __init__(self, seed, P=2038074743, **kwargs):
        super(UHasher, self).__init__(seed)
        self.P = P
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        print("Hashing using seed", seed)

        self.random_numbers = torch.randint(low=1, high=int(self.P/2) - 1, size=(4,), generator=self.gen)
        self.random_numbers = 2*self.random_numbers + 1
        

    def hash1(self, numbers, target_size=None):
        assert(target_size < self.P)
        self.random_numbers = self.random_numbers.to(numbers.device)
        return ((numbers * self.random_numbers[0] + torch.square(numbers) * self.random_numbers[1] + self.random_numbers[2]) % self.P) % target_size
        #return ((numbers * self.random_numbers[0] +  self.random_numbers[1]) % self.P) % target_size

    def hash2(self, number1, number2, target_size=None):
        assert(target_size < self.P)
        self.random_numbers = self.random_numbers.to(number1.device)
        return ((number1 * self.random_numbers[0] + number2 * self.random_numbers[1] + self.random_numbers[2]) % self.P) % target_size

class HasherFactory:
    def get(hasher, seed, **kwargs):
        if hasher == "uhash":
            return UHasher(seed, **kwargs)
        if hasher == "mhash":
            return MurmurHasher(seed, **kwargs)
        raise NotImplementedError


''' different mapping mechanisms '''
''' implements the simplest HashedNet mapping using uhash '''
class Mapper:
    def __init__(self, **kwargs):
        pass

    def get_general_idx(self, w_shape, original_offset, target_size, **kwargs):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num) + original_offset
      idx = (self.hasher.hash1(global_locations, target_size)) % target_size
      idx = idx.reshape(*w_shape)
      return idx

    def get_mlp_idx(self, **kwargs):
        return self.get_general_idx(**kwargs)

    def get_embedding_idx(self, **kwargs):
        return self.get_general_idx(**kwargs)

    def get_general_g(self, w_shape, original_offset, target_size, **kwargs):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num) + original_offset
      g = 2*(self.hasher.hash1(global_locations, 2)) - 1
      g = g.reshape(*w_shape)
      #print("Mapper get_general_idx")
      #print(idx[:5,:5])
      return g

    
    def get_mlp_g(self, **kwargs):
        return self.get_general_g(**kwargs)

    def get_embedding_g(self, **kwargs):
        return self.get_general_g(**kwargs)

    def get_idx(self, mode, **kwargs):
        if mode == "mlp":
            return self.get_mlp_idx(**kwargs)
        if mode == "embedding":
            return self.get_embedding_idx(**kwargs)

        return self.get_general_idx(**kwargs)

    def get_g(self, mode, **kwargs):
        if mode == "mlp":
            return self.get_mlp_g(**kwargs)
        if mode == "embedding":
            return self.get_embedding_g(**kwargs)

        return self.get_general_g(**kwargs)

class HashedNetMapper(Mapper):
    def __init__(self, hasher, **kwargs):
      super(HashedNetMapper, self).__init__()
      self.hasher = HasherFactory.get(hasher, **kwargs)
    
    def get_general_idx(self, w_shape, original_offset, target_size, **kwargs):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num) + original_offset
      idx = (self.hasher.hash1(global_locations, target_size)) % target_size
      idx = idx.reshape(*w_shape)
      #print("HashedNetMapper get_general_idx")
      #print(idx[:5,:5])
      return idx
        


class ParetoCyclicMapper(Mapper):
    def __init__(self, hasher, **kwargs):
      super(ParetoCyclicMapper, self).__init__()
      self.hasher = HasherFactory.get(hasher, **kwargs)
    
    def get_general_idx(self, w_shape, original_offset, target_size, **kwargs):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num) + original_offset
      chunk_num = global_locations // target_size
      offsets = global_locations - target_size * chunk_num
      idx = (self.hasher.hash1(chunk_num, target_size) + offsets) % target_size
      idx = idx.reshape(*w_shape)
      #print("ParetoCyclicMapper get_general_idx")
      #print(idx[:5,:5])
      return idx

class RoastMapper(Mapper):
    def __init__(self, hasher, **kwargs):
      super(RoastMapper, self).__init__()
      self.hasher = HasherFactory.get(hasher, **kwargs)
    
    def get_mlp_idx(self, w_shape, original_offset, target_size, block_k, block_n, **kwargs):
      assert(len(w_shape) == 2)
      row_chunk = torch.arange(w_shape[0]).reshape(-1,1) // block_k + original_offset
      col_chunk = torch.arange(w_shape[1]).reshape(1,-1) // block_n + original_offset +  w_shape[0]

      chunk_locations = self.hasher.hash2(row_chunk, col_chunk, target_size - block_k * block_n)

      offset = torch.arange(block_k*block_n).reshape(block_k, block_n).repeat(
                          (w_shape[0] + block_k) // block_k, (w_shape[1] + block_n) // block_n
                        )[:w_shape[0], :w_shape[1]]
      idx = chunk_locations + offset
      #print("RoastMapper get_mlp_idx")
      #print(idx[:5,:5])
      return idx


    def get_embedding_idx(self, w_shape, original_offset, target_size, block, **kwargs):
      assert(len(w_shape) == 2)
      row_num = torch.arange(w_shape[0]).reshape(-1,1) +  original_offset
      col_chunk = torch.arange(w_shape[1]).reshape(1,-1) // block + original_offset +  w_shape[0]

      chunk_locations = self.hasher.hash2(row_num, col_chunk, target_size - block)

      offset = torch.arange(block).reshape(1, block).repeat(
                          w_shape[0], (w_shape[1] + block) // block
                        )[:w_shape[0], :w_shape[1]]
      idx = chunk_locations + offset

      #print("RoastMapper get_embedding_idx")
      #print(idx[:5,:5])
      return idx




class RoastMemOptMapper(RoastMapper):
    def __init__(self, hasher, **kwargs):
      super(RoastMemOptMapper, self).__init__(hasher, **kwargs)
    
    def get_mlp_idx(self, w_shape, original_offset, target_size, block_k, block_n, **kwargs):
      assert(len(w_shape) == 2)
      col_chunk = torch.arange(w_shape[1]).reshape(1,-1) // block_n + original_offset + w_shape[0]
      row_chunk = torch.zeros(w_shape[0], dtype=torch.int64).reshape(-1,1) + original_offset# same column
      chunk_locations = self.hasher.hash2(row_chunk, col_chunk, target_size - block_k * block_n)
      # every chunk_row will have permutations
      rows = []
      for i in range((w_shape[0] + block_k) // block_k):
          column = []
          for j in range((w_shape[1] + block_n) // block_n):
              block = torch.randperm(block_k*block_n).reshape(block_k, block_n)
              column.append(block)
          rows.append(torch.cat(column, axis=1))

      offset = torch.cat(rows, axis=0)[:w_shape[0], :w_shape[1]]
      idx = chunk_locations + offset
      #print("RoastMemOptMapper get_mlp_idx")
      #print(idx[:5,:5])
      return idx


class RoastCompOptMapper(RoastMapper):
    def __init__(self, hasher, **kwargs):
      super(RoastCompOptMapper, self).__init__(hasher, **kwargs)
    
    def get_mlp_idx(self, w_shape, original_offset, target_size, block_k, block_n, block_k_small, **kwargs):
      assert(len(w_shape) == 2)

      row_chunk = torch.arange(w_shape[0]).reshape(-1,1) // block_k + original_offset
      col_chunk = torch.arange(w_shape[1]).reshape(1,-1) // block_n + original_offset + w_shape[0]

      chunk_locations = self.hasher.hash2(row_chunk, col_chunk, target_size - block_k * block_n)
      # every chunk_row will have permutations
      rows = []
      for i in range((w_shape[0] + block_k) // block_k):
          column = []
          for j in range((w_shape[1] + block_n) // block_n):
              index_array = torch.arange(block_k_small).repeat(((block_k + block_k_small) // block_k_small))[torch.randperm(block_k)]
              block = torch.randperm(block_k_small*block_n).reshape(block_k_small, block_n)[index_array]
              column.append(block)
          rows.append(torch.cat(column, axis=1))

      offset = torch.cat(rows, axis=0)[:w_shape[0], :w_shape[1]]
      idx = chunk_locations + offset
      #print("RoastCompOptMapper get_mlp_idx")
      #print(idx[:5,:5])
      return idx


class MapperFactory:
    def get(mapper, hasher, seed, **kwargs):
        if mapper == "hashednet":
            return HashedNetMapper(hasher=hasher, seed=seed, **kwargs)
        if mapper == "pareto":
            return ParetoCyclicMapper(hasher=hasher, seed=seed, **kwargs)
        if mapper == "roast":
            return RoastMapper(hasher=hasher, seed=seed, **kwargs)
        if mapper == "roast_mem":
            return RoastMemOptMapper(hasher=hasher, seed=seed, **kwargs)
        if mapper == "roast_comp":
            return RoastCompOptMapper(hasher=hasher, seed=seed, **kwargs)
        raise NotImplementedError
