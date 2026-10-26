# ARM wheel 压缩分卷

这里保存 Python 3.11 / 3.12 两组原始 wheel 的 XZ 压缩分卷，每卷不超过 768 KiB。整文件推送曾发生连接中断，因此使用小分卷逐个推送。

克隆本分支后，在仓库根目录执行：

```bash
python3 wheel_archives/restore.py
sha256sum -c WHEEL_SHA256SUMS
```

四个原始 wheel 会还原到 `wheels/`。脚本先校验每卷和完整压缩包，再校验还原后的 wheel SHA256；不会修改 wheel 内的任何文件。只需要一个 Python 版本时可加 `--abi cp311` 或 `--abi cp312`。

分卷顺序、大小、压缩包和原始 wheel 的 SHA256 全部保存在 `manifest.json`。无需安装额外 Python 库，使用标准库 `lzma`。
