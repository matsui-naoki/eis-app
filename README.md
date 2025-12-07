# EIS Analyzer

**Electrochemical Impedance Spectroscopy (EIS) Analysis Web Application**

Python Streamlit製のインピーダンス解析Webアプリケーション（β版）

---

## 特徴

- **データ読み込み**: BioLogic (.mpt) と ZPlot (.z) 形式のインピーダンスファイルをサポート
- **インタラクティブな可視化**: Plotlyによる対話的なNyquist線図、Bode線図、Arrhenius線図
- **等価回路解析**: pyEIS表記の等価回路モデルによるフィッティング
- **多点解析**: 複数温度でのデータを一括解析し、導電率を計算
- **マッピング解析**: 空間分布・組成分布の可視化（1D, 2D, Ternary）
- **セッション保存**: 解析結果をJSON形式で保存・読み込み

---

## インストール

### 1. 依存パッケージのインストール

```bash
cd eisanalyzer
pip install -r requirements.txt
```

### 2. オプション機能のインストール

クラスタリング機能を使用する場合：
```bash
pip install umap-learn
```

ブラックボックス最適化を使用する場合：
```bash
pip install optuna timeout-decorator
```

---

## 使用方法

### アプリの起動

```bash
cd eisanalyzer
streamlit run app.py
```

ブラウザが自動的に開き、アプリケーションが表示されます。

### 基本的なワークフロー

1. **サンプル情報の入力**
   - 左サイドバー上部で、サンプル名、厚さ、直径（または面積）を入力
   - Arrheniusモードを有効にする場合はトグルスイッチをON

2. **データファイルのアップロード**
   - サイドバーの「📁 Files」タブを選択
   - 「Upload EIS Data Files」からファイルを選択（複数可）
   - ドラッグ&ドロップまたはファイル選択画面から読み込み

3. **データの確認**
   - サイドバーの「📊 Data」タブでファイルを選択
   - Frequency, Z', Z'' のテーブルが表示される
   - Arrheniusモードが有効な場合、温度を入力

4. **プロットの表示**
   - メインパネルの「📈 Plots」タブを選択
   - プロットタイプ（Nyquist/Bode/Arrhenius）を選択
   - 表示するファイルを選択

5. **等価回路解析**
   - 「🔬 Circuit Analysis」タブを選択
   - サイドバーで解析するファイルを選択
   - 等価回路モデルと初期値を入力
   - 「Fit Circuit」ボタンをクリック
   - フィッティング結果が表示される

6. **多点解析**
   - 「📋 Multipoint Table」タブで全ファイルの解析結果を一覧表示
   - CSVファイルとしてダウンロード可能

7. **マッピング解析**（Mapping Mode）
   - 空間分布や組成分布の可視化
   - 1D: 位置 vs 伝導率の折れ線/散布図
   - 2D: ヒートマップ/等高線図（補間オプション付き）
   - Ternary: 3成分組成図（三角ダイアグラム）
   - Settingsでカラースケール、軸範囲、ラベル等をカスタマイズ可能

8. **セッションの保存**
   - サイドバー上部の「Save Session」ボタンをクリック
   - JSONファイルがダウンロードされる

---

## 等価回路表記

pyEIS表記に準拠した等価回路モデルを使用します。

### 記法例

- `p(R1,CPE1)`: R1とCPE1の並列回路
- `p(R1,CPE1)-CPE2`: R1とCPE1の並列回路に、CPE2を直列接続
- `p(R1,CPE1)-p(R2,CPE2)-CPE3`: 2つのRC並列回路とスパイクCPE

### パラメータ

- `R`: 抵抗（Ω）
- `CPE`: Constant Phase Element (Q, n)
  - Q: 疑似容量（F·s^(n-1)）
  - n: 指数（0 < n ≤ 1）

### 初期値の例

```
p(R1,CPE1)-CPE2
初期値: 1e6, 1e-9, 0.9, 1e-6, 0.9
```

- R1 = 1e6 Ω
- CPE1_Q = 1e-9
- CPE1_n = 0.9
- CPE2_Q = 1e-6
- CPE2_n = 0.9

---

## プロジェクト構造

```
eisanalyzer/
├── app.py                      # メインアプリケーション
├── eis.py                      # EIS解析ユーティリティ
├── requirements.txt            # 依存パッケージ
├── README.md                   # このファイル
│
├── components/                 # UIコンポーネント
│   ├── __init__.py
│   └── plots.py                # Plotly可視化
│
├── tools/                      # ツールモジュール
│   ├── __init__.py
│   └── data_loader.py          # データ読み込み
│
└── pyeis/                      # pyEISコアライブラリ
    ├── __init__.py
    ├── preprocessing.py        # データ前処理
    ├── visualization.py        # matplotlib可視化
    └── models/
        └── circuits/           # 等価回路モデル
            ├── circuits.py
            ├── elements.py
            └── fitting.py
```

---

## トラブルシューティング

### インポートエラーが発生する

依存パッケージが正しくインストールされているか確認：
```bash
pip install -r requirements.txt
```

### ファイルの読み込みに失敗する

- ファイル形式が .mpt または .z であることを確認
- ファイル内容がBioLogicまたはZPlotの標準形式に準拠しているか確認

### フィッティングが収束しない

- 初期値を調整
- 重み付け方法を変更（None / modulus / proportional）
- 解析範囲（周波数範囲）を調整

---

## 開発履歴

- **v0.2 (Beta)**: マッピング機能追加
  - Mapping Mode（1D, 2D, Ternary）の追加
  - scipy補間による等高線プロット
  - 化学式の自動下付き文字変換（Li7 → Li₇）
  - 3モード切り替え（Nyquist / Arrhenius / Mapping）

- **v0.1 (Beta)**: 初版リリース
  - 基本的なデータ読み込みと可視化機能
  - 等価回路フィッティング
  - 多点解析とArrhenius線図

---

## ライセンス

このプロジェクトは研究・教育目的で使用できます。

---

## 謝辞

- pyEIS: https://github.com/ECSHackWeek/impedance.py からフォークして再編集
- Streamlit: https://streamlit.io/
- Plotly: https://plotly.com/python/

---

**Created with Claude Code**
