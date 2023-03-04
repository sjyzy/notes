# git_sourcetree



## 链接github

### 1.win生成密钥

```shell
ssh-keygen -t rsa -C “1328801779@qq.com”
```

生成密钥id_rsa和id_rsa.pub



### 2.sourcetree 工具->选项，

ssh客户端选择OpenSSH，选择id_rsa。



### 3.进入githu->setting->SSH and GPG key

New SSH key

Title随便写

用记事本打开id_rsa.pub，全部复制下来，粘贴到Key中

Add SSH key

