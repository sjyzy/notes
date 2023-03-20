# conda 指令

## 创建

```shell
conda create -n your_env_name python=x.x
```



## 删除

```shell
conda remove -n your_env_name --all
```



## 克隆

```shell
conda create -n BBB --clone AAA
```



## 配置清华源

```sh
#查看当前conda配置
conda config --show channels
 
#设置通道
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
 
#设置搜索是显示通道地址
conda config --set show_channel_urls yes
# pytorchan'zhua
conda install pytorch torchvision cudatoolkit=10.0  # 删除安装命令最后的 -c pytorch，才会采用清华源安装。

# 查看配置信息
conda config --show

# 恢复默认源
conda config --remove-key channels

# 删除旧镜像源
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/cpu/
```

## 打包环境

```sh
# 条件
conda install -c conda-forge conda-pack
# 打包
conda pack -n 虚拟环境名称 -o 目标虚拟环境名
# 在envs目录下
tar -xzvf output.tar.gz -C /anaconda(或者miniconda)/envs/创建的文件夹/
conda env list
```

