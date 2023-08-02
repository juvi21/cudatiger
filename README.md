# tritonizedTiger
An accelerated implementation of the Tiger optimizer for PyTorch, supercharged with Triton for enhanced CUDA GPU efficiency in under 100 lines of python/triton. 
Tiger is an extremely memory efficient Optimizer and also should be slightly faster than it's counterparts ADAM, SGD etc.

## TODO

- Add benchmarks comparing ADAM, TIGER, SGD, etc.
- Provide more examples.
- Introduce testing.
- Improve this README.
  
## Citations

```bibtex
@misc{tigeropt,
  title={Tiger: A Tight-fisted Optimizer},
  author={Jianlin Su},
  year={2023},
  howpublished={\url{https://github.com/bojone/tiger}},
}
```
```bibtex
@article{Tillet2019TritonAI,
    title   = {Triton: an intermediate language and compiler for tiled neural network computations},
    author  = {Philippe Tillet and H. Kung and D. Cox},
    journal = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year    = {2019}
}
```
