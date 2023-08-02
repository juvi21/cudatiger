# tritonizedTiger
<p align="center">
  <img src="assets/cute_tiger.png" width="300" height="300" alt="Cute Llama">
</p>

An accelerated implementation of the Tiger optimizer for PyTorch, supercharged with Triton for enhanced CUDA GPU efficiency in under 100 lines of python/triton. 
Tiger is an extremely memory efficient Optimizer and also should be slightly faster than it's counterparts ADAM, SGD etc. **Inspired by:** [bojone/tiger](https://github.com/bojone/tiger/tree/main)

## Comparison
<img src="assets/Tiger-Lion-AdamW.png" alt="form">

## TODO

- Add benchmarks comparing ADAM, TIGER, SGD, etc.
- Provide more examples.
- Introduce testing.
- Improve this README.
- Push pypi
  
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
## Art
@Midjourney

## License:
This project is licensed under the MIT License.
