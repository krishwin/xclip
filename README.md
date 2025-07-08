# xclip
This model is a fork of https://github.com/microsoft/VideoX/tree/master/X-CLIP but with below improvements
- Removed dependency with NVIDIA APEX replaced with native pytorch AMP
- upgraded MMCV dependency to more recent supported version
- Notebook to finetune XCLIP with Custom dataset in kaggle
- Added RepNet model to use XCLIP  as Base for repetition counting 
