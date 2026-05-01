# FSST VCHAR GPU Decompression Test

TPC-H ORDERS テーブルの O_COMMENT カラムを FSST 圧縮し、GPU 上で展開 + KMP スキャン (Q13 NOT LIKE '%special%requests%') を行うスタンドアロンテスト。

## ビルド

```bash
cd gsst/test
make
```

要件: CUDA Toolkit, g++ (C++17), NVIDIA GPU (sm_80 以上)。
AVX512 対応 CPU では自動検出され、CPU 側 FSST 展開ベンチマークで AVX512 gather が有効になる。

CUDA_ARCH を変更する場合:

```bash
make CUDA_ARCH="-gencode arch=compute_90,code=sm_90"
```

## 実行

```bash
./fsst_vchar_test <orders.tbl> [num_iterations] [num_threads] [--skip-cpu]
```

| 引数 | デフォルト | 説明 |
|---|---|---|
| `orders.tbl` | (必須) | 単一ファイルまたは split ファイルのベースパス |
| `num_iterations` | 10 | GPU カーネルの繰り返し回数 |
| `num_threads` | CPU コア数 | パース・ページ構築の並列スレッド数 |
| `--skip-cpu` | (off) | CPU 展開スループットベンチマークをスキップ |

入力は単一ファイル (`orders.tbl`) と split ファイル (`orders.tbl.1`, `orders.tbl.2`, ...) の両方に対応。
単一ファイルは mmap + マルチスレッドパース、split ファイルはファイル単位で並列読み込みする。

### 実行例

```bash
# SF1 (split files)
./fsst_vchar_test /export/data1/tpch/input1/orders/orders.tbl

# SF10 (single file, 3 iterations, 24 threads)
./fsst_vchar_test /export/data1/tpch/orders_sf10.tbl 3 24

# SF100 (single file, 1 iteration, skip CPU benchmarks)
./fsst_vchar_test /export/data1/tpch/orders.tbl 1 --skip-cpu
```

## テスト内容

1. **CPU reference (raw KMP)** -- 非圧縮文字列での KMP マッチ (正解値)
2. **CPU reference (FSST decomp + KMP)** -- FSST 展開後の KMP マッチ (圧縮・展開の正当性検証)
3. **GPU Test 1: Uncompressed KMP Scan** -- 非圧縮データの GPU KMP スキャン (ベースラインスループット)
4. **GPU Test 2: FSST Decomp + KMP Scan** -- FSST 展開 + KMP の fused カーネル
5. **CPU FSST Decompress Bandwidth** -- CPU 展開スループット (`--skip-cpu` でスキップ可能)
6. **GPU Memory-Level Comparison** -- Global memory / Cache / Coalesced writeback の比較
7. **Decompress-Only Kernel: Comp Block Size Sweep** -- comp block サイズ別の展開スループット
8. **Fused Decomp+KMP Kernel: Comp Block Size Sweep** -- comp block サイズ別の fused カーネルスループット

## ページフォーマット

```
+-------------------------------------+
| pag_head (12B)                      |
|   nalloc, watermark, lfreespace     |
+-------------------------------------+
| uint32_t n_comp_blocks (4B)         |
+-------------------------------------+
| FsstCompBlockDirEntry[n_comp_blocks]|
|   (8B each: offset, nrecs)         |
+-------------------------------------+
| FSST Symbol Table (2296B)           |
|   len[256] + symbol[255] (u64x255) |
+-------------------------------------+
| Comp Block 0:                       |
|   uint16_t offset_table[nrecs+1]   |
|   compressed records (packed)       |
+-------------------------------------+
| Comp Block 1: ...                   |
+-------------------------------------+
```

## ファイル構成

| ファイル | 内容 |
|---|---|
| `fsst_vchar_test.cu` | メインプログラム (パース, GPU カーネル, ベンチマーク) |
| `fsst_page.h` | ページフォーマット定義 (pag_head, FsstCompBlockDirEntry, FsstCompBlockKernelMeta) |
| `fsst_host.h / .cpp` | CPU 側 API (fsst_serialize_symbol_table, pagcol_append_batch_unordered_column_vchar_fsst) |
| `fsst_device.cuh` | GPU カーネル (decompress_string_with_fsst, decompress_scan_string_with_fsst) |
| `cpu_fsst_avx512.h / .cpp` | CPU AVX512 FSST 展開 |
| `Makefile` | ビルド設定 |
