# clash设置



## 预处理文件

```json
parsers: # array
  - url: https://getnode.dksb.store/subscribe/123288/RkFMB34zxfbG
    yaml:
      prepend-rules:
        - DOMAIN-SUFFIX,chat.openai.com,GLOBAL


```

```
#规则类型选择
DOMAIN-SUFFIX：域名后缀匹配
DOMAIN：域名匹配
DOMAIN-KEYWORD：域名关键字匹配
IP-CIDR：IP段匹配
SRC-IP-CIDR：源IP段匹配
GEOIP：GEOIP数据库（国家代码）匹配
DST-PORT：目标端口匹配
SRC-PORT：源端口匹配
PROCESS-NAME：源进程名匹配
RULE-SET：Rule Provider规则匹配
MATCH：全匹配
# 代理模式
DIRECT表示不走代理，即不通过代理节点直接连接。
GLOBAL则是走全局代理节点。
REJECT则表示禁止连接，使用REJECT后，将会屏蔽对应网站。
```

