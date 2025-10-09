# screen基本用法
## 1. 启动新的screen会话:

```shell
 \#创建名为name的会话
screen -S name 

\# 创建一个会话, 系统自动命名(形如:XXXX.pts-53.ubuntu)
screen
```

## 2. 退出当前screen会话:

```shell
按:Ctrl+a, 再按:d, 即可退出screen, 此时,程序仍在后台执行;
```

## 3. 查看当前已有的screen会话:

```shell
输入:screen -ls;
```

## 4.进入某个会话:

```shell
输入:screen -r 程序进程ID, 返回程序执行进程;
```

## 5. 窗口操作:

```shell
Ctrl+a+w: 展示当前会话中的所有窗口;
Ctrl+a+c: 创建新窗口;
Ctrl+a+n: 切换至下一个窗口;
Ctrl+a+p: 切换至上一个窗口;
Ctrl+a+num: 切换至编号为num的窗口;
Ctrl+a+k: 杀死当前窗口;
```

## 6. 删除某个会话:

```
screen -S your_screen_name -X quit
```

## 7.screenrc:

```vim
caption always "%{= kw}%-w%{= kG}%{+b}[%n %t]%{-b}%{= kw}%+w %= %{g}%H%{-}"
defscrollback 1000
# Set screen buffer
defscrollback 40000

# Enable mouse scrolling and scroll bar history scrolling
termcapinfo xterm* ti@:te@

# Enable status bar and display session name
hardstatus on
hardstatus alwayslastline
hardstatus string "%S"
```

