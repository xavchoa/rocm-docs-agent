# Candidate eval questions from real users

Collected 2026-11-14 from search results (GitHub issues, Reddit r/ROCm, forums). Questions
are paraphrased from thread titles and search snippets. A thread may ask something slightly
different, and old threads refer to old ROCm versions (the corpus is ROCm 10.0.0, `develop`
at the commit in data/raw/rocm/SOURCE.txt).

"Docs mention" = pages in data/clean/rocm/ where the key terms appear (grep). It means the
page *might* answer the question, not that it does. Expected answers are not in this file:
they were checked against the pages and written to evals/questions.jsonl, which holds the
25 questions picked from this list.

## Install and setup

| # | Question (paraphrased) | Source | Docs mention |
| --- | --- | --- | --- |
| 1 | rocminfo says "Unable to open /dev/kfd read-write: Permission denied". What's wrong? | [openbenchmarking rocminfo log](https://openbenchmarking.org/system/2304054-PTS-AMDRYZEN15/a/rocminfo) | install/rocm.md (render,video groups); the exact error text is not in the docs |
| 2 | I followed the install steps but it still doesn't work; what did I miss? (general install help) | [r/ROCm: Installation help](https://www.reddit.com/r/ROCm/comments/1j3eyp5/installation_help) | install/rocm.md (open the thread to get the specific problem) |
| 3 | Can I install two ROCm versions side by side? | [rocminstaller (unofficial project)](https://gh.hhj520.top/sidamd/rocminstaller) | about/transition-guide-TheRock.md ("multiple ROCm versions side by side" -> Tarball) |
| 4 | How do I uninstall ROCm completely? | [rocminstaller (unofficial project)](https://gh.hhj520.top/sidamd/rocminstaller) | install/rocm.md ("## Uninstalling") |
| 5 | Which install method should I use: package manager, pip or tarball? | [TheRock issue #2110](https://github.com/ROCm/TheRock/issues/2110) | install/rocm.md, about/transition-guide-TheRock.md |
| 6 | pip install keeps giving me an old PyTorch build for my GPU target. Which index should I use? | [TheRock issue #2110](https://github.com/ROCm/TheRock/issues/2110) | about/transition-guide-TheRock.md, install/rocm.md (index-url) |
| 7 | What is TheRock and how is it different from the old ROCm releases? | [TheRock issue #2110](https://github.com/ROCm/TheRock/issues/2110) | about/transition-guide-TheRock.md, about/what-is-rocm.md |

## GPU support and product names (tests option D / C)

| # | Question (paraphrased) | Source | Docs mention |
| --- | --- | --- | --- |
| 8 | Is the RX 7900 XTX officially supported, or do I still need workarounds? | [dev.to article on PyTorch with the 7900 XTX](https://dev.to/pickuma/rocm-in-2026-why-pytorch-on-the-rx-7900-xtx-still-falls-short-for-research-36pe) | product name only in about/transition-guide-TheRock.md; gfx1100 in install/rocm.md |
| 9 | Does ROCm work with an RX 6800? | [r/ROCm: ROCm compatibility with RX6800](https://www.reddit.com/r/ROCm/comments/1j1eio9/rocm_compatibility_with_rx6800) | "RX 6800" is in no page and not in gpus.yaml; gfx1030 appears in install/rocm.md. A good "not listed" test |
| 10 | I got ROCm working on a 7600 XT. Is that card actually supported? | [r/ROCm: I made ROCm work with 7600XT](https://www.reddit.com/r/ROCm/comments/1o1bnc1/i_made_rocm_work_with_7600xt/) | "7600 XT" in no page; check gpus.yaml |
| 11 | Does the Ryzen AI Max 365 support ROCm? | [r/ROCm: Does Ryzen AI Max 365 support ROCm](https://www.reddit.com/r/ROCm/comments/1k94sk1/does_ryzen_ai_max_365_support_rocm) | install/rocm.md, about/transition-guide-TheRock.md (gfx1151), reference/system-optimization/rdna3-5.md |
| 12 | Strix Halo (gfx1151): are there kernel requirements? | [Strix Halo knowledge base (community)](https://whylucian.github.io/softab/KNOWLEDGE_BASE.html) | reference/system-optimization/rdna3-5.md (Ubuntu 26.04, HWE kernel) |
| 13 | How do I give the Strix Halo iGPU more memory? | [Strix Halo knowledge base (community)](https://whylucian.github.io/softab/KNOWLEDGE_BASE.html) | reference/system-optimization/rdna3-5.md (TTM page limit) |
| 14 | My GPU isn't supported. Should I set HSA_OVERRIDE_GFX_VERSION? | [ROCm/ROCm issue #2527](https://github.com/ROCm/ROCm/issues/2527) | HSA_OVERRIDE_GFX_VERSION is in no page. A good out-of-scope or "not in the docs" test |
| 15 | Does HSA_OVERRIDE_GFX_VERSION work on Windows? | [r/ROCm: Out of luck on HIP SDK](https://www.reddit.com/r/ROCm/comments/1jmvn5a/out_of_luck_on_hip_sdk) | not in the docs |

## Windows and WSL

| # | Question (paraphrased) | Source | Docs mention |
| --- | --- | --- | --- |
| 16 | Does ROCm really work under WSL2? | [r/ROCm: Does ROCm really work with WSL2?](https://www.reddit.com/r/ROCm/comments/1j2ns4g/does_rocm_really_work_with_wsl2) | install/rocm.md (os=wsl section: WSL2 + Ubuntu 26.04) |
| 17 | ROCm doesn't detect my GPU in WSL2. | [r/ROCm: Help with ROCm and WSL2](https://www.reddit.com/r/ROCm/comments/1jxvw44/help_with_rocm_and_wsl2) | install/rocm.md (setup only; troubleshooting probably not covered) |
| 18 | Can I use PyTorch with ROCm on native Windows? | [r/ROCm: Windows PyTorch wheels](https://www.reddit.com/r/ROCm/comments/1l7uv0o/github_user_scottt_has_created_windows_pytorch) | install/rocm.md (os=windows), about/release-notes.md |
| 19 | How do I install ROCm on Windows 11? | [r/ROCm: Installation guide Windows 11](https://www.reddit.com/r/ROCm/comments/1n1jwh3/installation_guide_windows_11_rocm_7_rc_with) | install/rocm.md ("start with ... tarball on Windows") |
| 20 | Which GPUs are supported by the HIP SDK on Windows? | [r/ROCm: Out of luck on HIP SDK](https://www.reddit.com/r/ROCm/comments/1jmvn5a/out_of_luck_on_hip_sdk) | install/rocm.md, compatibility page is templated (data in YAML) |

## Frameworks and libraries

| # | Question (paraphrased) | Source | Docs mention |
| --- | --- | --- | --- |
| 21 | ComfyUI warns "attempting to use hipBLASLt on an unsupported architecture". What does it mean? | [PyTorch forum](https://discuss.pytorch.org/t/comfy-ui-attempting-to-use-hipblaslt-on-a-unsupported-architecture/215776) | hipBLASLt named in several pages; the warning itself is probably not covered |
| 22 | How do I set up PyTorch with ROCm for Stable Diffusion on a Framework laptop? | [Framework community](https://community.frame.work/t/stable-diffusion-rocm-pytorch-setup/53802) | ai-ecosystem.md, install/rocm.md |
| 23 | Which PyTorch, JAX and vLLM versions work with this ROCm release? | [TheRock issue #2110](https://github.com/ROCm/TheRock/issues/2110) | ai-ecosystem.md, about/release-notes.md; compatibility page templated |
| 24 | How do I port my CUDA code to AMD GPUs? | (common question; no single thread found) | reference/hip-programming.md (HIPIFY) |

## Notes

- Every source above came from search results; none of the threads were opened. The
  Reddit links are third-party mirrors because reddit.com itself did not come up.
- 9, 10, 14 and 15 are likely "not in the docs" in your corpus, which makes them good
  real out-of-scope questions (`expected_source: "none"`), unless gpus.yaml says otherwise.
- 8 to 13 test the product-name problem (option D vs baseline).
- Missing categories you may want to write yourself: pure concept questions ("what is
  ROCm?"), and compatibility questions whose answer is only in the YAML.
