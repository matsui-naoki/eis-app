# EIS Analyzer - クイックスタートガイド

## 5分で始めるEIS解析

### ステップ1: インストール

```bash
cd eisanalyzer
pip install -r requirements.txt
```

### ステップ2: アプリ起動

```bash
# 方法1: シェルスクリプト（推奨）
./run.sh

# 方法2: 直接実行
streamlit run app.py
```

ブラウザが自動的に開きます（通常 http://localhost:8501）

### ステップ3: サンプル情報を入力

左サイドバーの上部で：
- **Sample Name**: 任意の名前（例: "Sample A"）
- **Thickness**: 試料厚さ（例: 0.1 cm）
- **Diameter**: 試料直径（例: 1.0 cm）

### ステップ4: データをアップロード

1. サイドバーで「📁 Files」タブを選択
2. 「Upload EIS Data Files」をクリック
3. .mpt または .z ファイルを選択
4. 複数ファイルを同時に選択可能

### ステップ5: プロットを表示

1. メインパネルで「📈 Plots」タブを選択
2. Plot Type で「Nyquist」を選択
3. インタラクティブなNyquist線図が表示されます

### ステップ6: 等価回路フィッティング

1. 「🔬 Circuit Analysis」タブを選択
2. サイドバーで解析したいファイルをクリック
3. Circuit String を入力（例: `p(R1,CPE1)-CPE2`）
4. Initial Guess を入力（例: `1e6, 1e-9, 0.9, 1e-6, 0.9`）
5. 「Fit Circuit」ボタンをクリック
6. フィッティング結果が右側に表示されます

### ステップ7: 結果を確認

**フィッティング結果**:
- パラメータ値（R1, CPE1, etc.）
- RMSPE（フィッティングエラー）
- 導電率 σ (S/cm)

**プロットの更新**:
- 「📈 Plots」タブに戻る
- 「Show Fitted Data」をチェック
- フィッティング曲線が重ねて表示されます

### ステップ8: 結果を保存

**セッション保存**:
1. サイドバー上部の「Save Session」をクリック
2. JSONファイルがダウンロードされます

**CSV エクスポート**:
1. 「📋 Multipoint Table」タブを選択
2. 「Download CSV」をクリック

---

## 多温度データ解析（Arrheniusプロット）

### ステップ1: Arrheniusモードを有効化

サイドバー上部の「Arrhenius Mode」チェックボックスをON

### ステップ2: 各ファイルに温度を設定

1. サイドバーで「📊 Data」タブを選択
2. ファイルを選択
3. 「Temperature (K)」に温度を入力（例: 298.15）
4. 別のファイルを選択して同様に温度を設定

### ステップ3: 各ファイルをフィッティング

各温度データに対して等価回路フィッティングを実行

### ステップ4: Arrheniusプロットを表示

1. 「📈 Plots」タブを選択
2. Plot Type で「Arrhenius」を選択
3. Conductivity Type を選択（total/bulk/gb）
4. log(σT) vs 1000/T のプロットが表示されます

### ステップ5: 活性化エネルギーを計算

Arrheniusプロットの傾きから活性化エネルギーを計算：
```
Ea = -slope × k × ln(10)
```
- k = 8.617 × 10⁻⁵ eV/K（ボルツマン定数）

---

## よくある質問

### Q1: フィッティングが収束しない

**A**: 以下を試してください：
- 初期値を変更（特にRとCPEの値）
- Weight Method を変更（None → modulus → proportional）
- 回路モデルを簡単なものに変更

### Q2: ファイルが読み込めない

**A**: ファイル形式を確認してください：
- 対応形式: .mpt (BioLogic), .z (ZPlot)
- ファイルが破損していないか確認
- ファイルサイズが大きすぎないか確認（< 200MB）

### Q3: プロットがおかしい

**A**: 以下を確認してください：
- データの品質（ノイズ、異常値）
- 周波数範囲が適切か
- インピーダンスの値が妥当な範囲か

---

## サンプルデータ

サンプルデータは以下のような形式です：

**BioLogic .mpt**:
```
freq/Hz    Re(Z)/Ω    -Im(Z)/Ω
1000000    100.5       -5.2
100000     105.3       -10.8
...
```

**ZPlot .z**:
```
Frequency    Z'         Z''
1e6          100.5      5.2
1e5          105.3      10.8
...
```

---

## トラブルシューティング

### エラー: "No module named 'streamlit'"

```bash
pip install streamlit
```

### エラー: "No module named 'plotly'"

```bash
pip install -r requirements.txt
```

### エラー: "Port 8501 is already in use"

別のStreamlitアプリが起動中です：
```bash
# 別のポートで起動
streamlit run app.py --server.port 8502
```

---

## 次のステップ

- [README.md](README.md) - 詳細な機能説明
- [SPECIFICATION.md](SPECIFICATION.md) - 技術仕様書
- GitHub Issues - バグ報告・機能要望

---

**Happy Analyzing! ⚡**
