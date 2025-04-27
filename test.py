import os
import torch
import torch.distributed as dist

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')  # 使用 NCCL 后端（GPU），或改为 'gloo'（CPU）

    # 获取当前进程的 rank 和 world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 打印基本信息
    print(f"Rank {rank}/{world_size} initialized successfully.")

    # 简单的 all-reduce 测试
    # 每个进程创建一个张量，内容为其 rank 值
    tensor = torch.tensor(float(rank)).cuda()  # 如果用 CPU，移除 .cuda()
    print(f"Rank {rank} initial tensor: {tensor.item()}")

    # 执行 all-reduce 操作，将所有进程的张量求和
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 验证结果：对于两台机器，world_size=2，tensor 应为 0+1=1
    print(f"Rank {rank} after all-reduce: {tensor.item()}")

    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()