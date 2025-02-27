# MT-DeepEP
This repository is forked from the open source project DeepEP deepseek-ai/DeepEP. It adapts DeepEP to work with Moore Threads GPUs.


## Quick start

### Requirements

- MooreThreads GPU (Compute Capability 3.1)
- Python 3.8 and above
- MUSA 4.0.0 and above
- PyTorch 2.1 and above
- MTLink for intranode communication
- RDMA network for internode communication

## License

This code repository is released under [the MIT License](LICENSE) 

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
