# EIS Analyzer - 詳細仕様書

Version: 0.1 (Beta)
Last Updated: 2025-12-03
Created with: Claude Code

---

## 1. 概要

### 1.1 プロジェクト概要
EIS Analyzerは、電気化学インピーダンス分光法（Electrochemical Impedance Spectroscopy）のデータを解析するためのWebアプリケーションです。Python Streamlitフレームワークを使用して構築されており、インタラクティブな可視化と等価回路フィッティング機能を提供します。

### 1.2 主要機能
1. インピーダンスデータファイルの読み込み（.mpt, .z形式）
2. Nyquist線図、Bode線図、Arrhenius線図の可視化
3. 等価回路モデルによるフィッティング
4. イオン伝導率の計算
5. 多点（多温度）解析
6. セッションデータの保存・読み込み

### 1.3 技術スタック
- **フロントエンド**: Streamlit 1.28+
- **可視化**: Plotly 5.17+, Matplotlib 3.7+
- **科学計算**: NumPy, SciPy, scikit-learn
- **データ処理**: Pandas
- **フィッティング**: pyEIS（カスタム実装）

---

## 2. アーキテクチャ

### 2.1 プロジェクト構造

```
eisanalyzer/
│
├── app.py                          # メインアプリケーション
│   - Streamlit UI
│   - セッション管理
│   - ページレイアウト
│
├── eis.py                          # EIS解析コアモジュール
│   - データ読み込み（load_eis, load_multi_eis）
│   - 等価回路フィッティング（auto_fit, fit）
│   - 導電率計算（r2sigma, r2logsigma）
│   - グラフ作成（make_graph）
│   - クラスタリング（optional）
│   - ブラックボックス最適化（optional）
│
├── components/                     # UIコンポーネント
│   ├── __init__.py
│   └── plots.py                    # Plotly可視化
│       - create_nyquist_plot()
│       - create_bode_plot()
│       - create_arrhenius_plot()
│
├── tools/                          # ユーティリティ
│   ├── __init__.py
│   └── data_loader.py              # データ読み込み
│       - load_uploaded_file()
│       - parse_biologic()
│       - parse_zplot()
│       - validate_impedance_data()
│
└── pyeis/                          # pyEISコアライブラリ
    ├── __init__.py
    ├── preprocessing.py            # データ前処理
    │   - readBioLogic()
    │   - readZPlot()
    │   - その他の形式
    │
    ├── visualization.py            # Matplotlib可視化
    │   - plot_nyquist()
    │   - plot_bode()
    │
    └── models/                     # モデル定義
        └── circuits/               # 等価回路
            ├── circuits.py         # CustomCircuit クラス
            ├── elements.py         # 回路素子（R, C, CPE, etc.）
            └── fitting.py          # フィッティングアルゴリズム
```

### 2.2 データフロー

```
[ファイルアップロード]
    ↓
[data_loader.load_uploaded_file()]
    ↓
[Session State: st.session_state.files]
    ↓
[可視化] ← → [等価回路解析]
    ↓              ↓
[Plotly]      [CustomCircuit.fit()]
    ↓              ↓
[表示]        [パラメータ抽出]
                   ↓
              [導電率計算]
                   ↓
            [多点解析テーブル]
                   ↓
            [Arrhenius線図]
```

---

## 3. 機能詳細

### 3.1 サイドバー機能

#### 3.1.1 サンプル情報入力（固定表示）
**場所**: サイドバー最上部

**入力項目**:
- サンプル名（テキスト入力）
- 厚さ / cm（数値入力、デフォルト: 0.1）
- 直径 / cm または 面積 / cm²（ラジオボタンで選択）
  - 直径モード: 直径から面積を計算
  - 面積モード: 面積を直接入力

**アクション**:
- Arrheniusモード（トグルスイッチ）
  - ON: 温度依存性解析を有効化
  - OFF: 単一温度での解析
- リセットボタン
  - 全セッションデータをクリア
- Save Sessionボタン
  - JSONファイルをダウンロード

**実装**: `sidebar_sample_info()` 関数

#### 3.1.2 ファイル管理タブ
**場所**: サイドバー下部（📁 Files タブ）

**機能**:
- ファイルアップローダー
  - 対応形式: .mpt (BioLogic), .z (ZPlot)
  - 複数ファイル同時アップロード可能
- ロード済みファイルリスト
  - ファイル名クリック → 選択
  - 📊 アイコン → データ表示
  - 🗑️ アイコン → ファイル削除

**実装**: `sidebar_file_manager()` 関数

#### 3.1.3 データビュータブ
**場所**: サイドバー下部（📊 Data タブ）

**表示内容**:
- 選択中のファイル名
- データテーブル
  - Frequency (Hz)
  - Z' (Ω) - 実部
  - Z'' (Ω) - 虚部

**Arrheniusモード時の追加入力**:
- Temperature (K) - 温度入力欄

**実装**: `sidebar_data_view()` 関数

#### 3.1.4 設定タブ
**場所**: サイドバー下部（⚙️ Settings タブ）

**現在のステータス**: 未実装（将来の拡張用）

**将来実装予定**:
- デフォルト回路モデル設定
- プロット設定（色、マーカーサイズ等）
- エクスポート設定

---

### 3.2 メインパネル機能

#### 3.2.1 Plotsタブ（📈 Plots）

**機能概要**: インピーダンスデータの可視化

**プロットオプション**:
- Show Fitted Data（チェックボックス）
  - ON: フィッティング結果を表示
  - OFF: 測定データのみ表示
- Plot Type（セレクトボックス）
  - Nyquist: Nyquist線図
  - Bode: Bode線図
  - Arrhenius: Arrhenius線図
- Select Files to Plot（マルチセレクト）
  - 表示するファイルを選択

**Nyquist線図**:
- 横軸: Z' (Ω)
- 縦軸: -Z'' (Ω)
- 等アスペクト比
- インタラクティブ（ズーム、パン可能）

**Bode線図**:
- 上段: |Z| vs Frequency（両対数）
- 下段: θ vs Frequency（θは線形、Freqは対数）

**Arrhenius線図**（Arrheniusモード時のみ）:
- 横軸: 1000/T (K⁻¹)
- 縦軸: log(σT) (S cm⁻¹ K)
- 導電率タイプ選択:
  - Total: 全導電率
  - Bulk: バルク導電率
  - GB: 粒界導電率

**実装**: `main_panel_plots()` 関数、`components/plots.py`

#### 3.2.2 Circuit Analysisタブ（🔬 Circuit Analysis）

**機能概要**: 等価回路モデルによるフィッティング

**左パネル: 回路モデル設定**
- Circuit String (pyEIS notation)
  - 等価回路モデルの文字列表記
  - 例: `p(R1,CPE1)-CPE2`
- Initial Guess
  - カンマ区切りの初期値リスト
  - 例: `1e6, 1e-9, 0.9, 1e-6, 0.9`
- Weight Method
  - None: 重み付けなし
  - modulus: インピーダンスの大きさで重み付け
  - proportional: 比例重み付け
  - error: エラーベース重み付け
- Fit Circuitボタン
  - フィッティング実行

**右パネル: フィッティング結果**
- パラメータテーブル
  - Parameter: パラメータ名
  - Value: フィッティング値
  - Unit: 単位（Ω, F, etc.）
- RMSPE: Root Mean Square Percentage Error
- Total Conductivity: 全抵抗から計算した導電率
- log(σ): 導電率の対数

**実装**: `circuit_analysis_panel()` 関数

#### 3.2.3 Multipoint Tableタブ（📋 Multipoint Table）

**機能概要**: 複数ファイルの解析結果一覧

**表示項目**:
- File: ファイル名
- Temperature (K): 温度
- 1000/T (K⁻¹): 逆温度
- 回路パラメータ（R1, CPE1_Q, CPE1_n, etc.）
- RMSPE: フィッティングエラー
- σ (S/cm): 導電率
- log(σ): 導電率の対数
- log(σT): 導電率×温度の対数（Arrheniusプロット用）

**アクション**:
- Download CSVボタン
  - テーブルをCSV形式でダウンロード

**実装**: `multipoint_analysis_table()` 関数

---

### 3.3 データ管理

#### 3.3.1 セッション状態

**st.session_state.files**:
```python
{
    'filename1.mpt': {
        'freq': np.ndarray,           # 周波数配列
        'Z': np.ndarray (complex),    # インピーダンス配列
        'Z_fit': np.ndarray (complex) or None,  # フィッティング結果
        'circuit_model': str or None,  # 回路モデル文字列
        'circuit_params': np.ndarray or None,  # フィッティングパラメータ
        'circuit_object': CustomCircuit or None,  # 回路オブジェクト
        'rmspe': float or None,        # フィッティングエラー
        'initial_guess': list or None, # 初期値
        'temperature': float or None,  # 温度 (K)
        'total_sigma': float or None,  # 全導電率
        'bulk_sigma': float or None,   # バルク導電率
        'gb_sigma': float or None      # 粒界導電率
    },
    ...
}
```

**st.session_state.sample_info**:
```python
{
    'name': str,         # サンプル名
    'thickness': float,  # 厚さ (cm)
    'diameter': float,   # 直径 (cm)
    'area': float       # 面積 (cm²)
}
```

**st.session_state.selected_file**: str or None
現在選択中のファイル名

**st.session_state.arrhenius_mode**: bool
Arrheniusモードの有効/無効

#### 3.3.2 セッション保存形式

**ファイル形式**: JSON

**構造**:
```json
{
  "timestamp": "2025-12-03T08:00:00",
  "sample_info": {
    "name": "Sample A",
    "thickness": 0.1,
    "diameter": 1.0,
    "area": 0.785
  },
  "files": {
    "data1.mpt": {
      "freq": [1e6, 1e5, ...],
      "Z_real": [100, 105, ...],
      "Z_imag": [-5, -10, ...],
      "Z_fit_real": [100.5, 104.8, ...],
      "Z_fit_imag": [-5.2, -9.8, ...],
      "circuit_model": "p(R1,CPE1)-CPE2",
      "circuit_params": [1e6, 1e-9, 0.9, 1e-6, 0.9],
      "initial_guess": [1e6, 1e-9, 0.9, 1e-6, 0.9],
      "rmspe": 0.0234,
      "temperature": 298.15,
      "total_sigma": 1.23e-5,
      "bulk_sigma": 2.34e-5,
      "gb_sigma": 5.67e-6
    }
  }
}
```

**実装**: `save_session()` 関数

---

## 4. 等価回路モデル

### 4.1 pyEIS表記

**基本記法**:
- `-`: 直列接続
- `p(A,B)`: AとBの並列接続
- `s(A,B)`: AとBの直列接続（明示的）

**回路素子**:
- `R`: 抵抗（Resistor）
- `C`: 容量（Capacitor）
- `CPE`: Constant Phase Element
- `L`: インダクタンス
- `W`: Warburg素子

### 4.2 CPEパラメータ

CPEのインピーダンス:
```
Z_CPE = 1 / (Q * (jω)^n)
```

パラメータ:
- `Q`: 疑似容量（F·s^(n-1)）
- `n`: 指数（0 < n ≤ 1）
  - n = 1: 理想容量
  - n = 0.5: Warburg拡散
  - n < 1: 非理想容量

### 4.3 一般的な等価回路モデル

**1. バルク抵抗 + スパイク**
```
p(R1,CPE1)-CPE2
```
- R1: バルク抵抗
- CPE1: バルク容量成分
- CPE2: 電極スパイク

**2. バルク + 粒界**
```
p(R1,CPE1)-p(R2,CPE2)-CPE3
```
- R1, CPE1: バルク
- R2, CPE2: 粒界
- CPE3: 電極スパイク

**3. 直列抵抗 + バルク + スパイク**
```
R0-p(R1,CPE1)-CPE2
```
- R0: 配線抵抗など
- R1, CPE1: バルク
- CPE2: スパイク

### 4.4 フィッティングアルゴリズム

**方法**: 非線形最小二乗法（scipy.optimize.least_squares）

**目的関数**:
```
minimize: Σ |Z_measured - Z_model|²
```

**重み付け**:
- None: 等重み
- modulus: w = 1/|Z|
- proportional: w = 1/(Re(Z)² + Im(Z)²)

**フィッティング手順**:
1. 初期値の設定
2. 重み付け方法1でフィッティング
3. 結果を初期値として、重み付け方法2でフィッティング
4. さらに重み付け方法3でフィッティング
5. 最終結果を返す

**実装**: `eis.fit()` 関数、`pyeis/models/circuits/fitting.py`

---

## 5. 導電率計算

### 5.1 計算式

**イオン伝導率**:
```
σ = L / (R × S)
```

- σ: イオン伝導率 (S/cm)
- L: 試料厚さ (cm)
- R: 抵抗 (Ω)
- S: 試料面積 (cm²)

**対数導電率**:
```
log(σ) = log10(σ)
```

**Arrhenius式のための変数**:
```
log(σT) = log10(σ × T)
```

- T: 絶対温度 (K)

### 5.2 実装

**関数**:
- `r2sigma(R, S, L)`: 導電率計算
- `r2logsigma(R, S, L)`: 対数導電率計算

**使用箇所**:
- Circuit Analysisタブでのフィッティング後
- Multipoint Tableでの一括表示

---

## 6. データ入出力

### 6.1 対応ファイル形式

#### 6.1.1 BioLogic .mpt形式

**特徴**:
- BioLogic EC-Lab ソフトウェアが生成
- タブ区切りテキスト形式
- ヘッダー行に "freq/Hz" を含む

**データ列**:
```
freq/Hz    Re(Z)/Ω    -Im(Z)/Ω    |Z|/Ω    Phase(Z)/deg    ...
```

**パーサー**: `tools/data_loader.parse_biologic()`

#### 6.1.2 ZPlot .z形式

**特徴**:
- Scribner Associates ZPlotソフトウェアが生成
- カンマまたはタブ区切り
- コメント行は '!' または '#' で始まる

**データ列**:
```
Frequency    Z'    Z''    ...
```

**パーサー**: `tools/data_loader.parse_zplot()`

### 6.2 セッション保存

**形式**: JSON

**保存内容**:
- サンプル情報
- 全ファイルのデータと解析結果
- タイムスタンプ

**ファイル名**: `eis_session_YYYYMMDD_HHMMSS.json`

**実装**: `save_session()` 関数

### 6.3 CSV エクスポート

**機能**: Multipoint Tableの内容をCSV形式でエクスポート

**ファイル名**: `eis_analysis_YYYYMMDD_HHMMSS.csv`

**実装**: Streamlit `st.download_button()` + pandas `to_csv()`

---

## 7. エラーハンドリング

### 7.1 ファイル読み込みエラー

**検証項目**:
- ファイル形式（.mpt, .z）
- データの有無
- 周波数と印ピーダンス配列の長さ一致
- 周波数値の正負
- インピーダンスのNaN/Inf チェック

**エラー表示**: サイドバーに `st.error()` メッセージ

### 7.2 フィッティングエラー

**発生原因**:
- 初期値が不適切
- 回路モデルが不適切
- データ品質が悪い

**エラー表示**: Circuit Analysisタブに `st.error()` メッセージ

**対処方法**:
- 初期値の調整
- 回路モデルの変更
- 重み付け方法の変更

### 7.3 インポートエラー

**オプショナル依存パッケージ**:
- `umap`: クラスタリング機能用（未使用でも動作）
- `optuna`: ブラックボックス最適化用（未使用でも動作）
- `timeout-decorator`: タイムアウト処理用（未使用でも動作）

**実装**: try-except でインポートし、利用不可の場合は機能を無効化

---

## 8. パフォーマンス最適化

### 8.1 データキャッシング

**Streamlit session_state**:
- ファイルデータをメモリに保持
- ページ遷移してもデータ維持

### 8.2 プロット最適化

**Plotly**:
- インタラクティブプロットはWebGLレンダリング
- 大量データ点の場合、サンプリング推奨

### 8.3 フィッティング最適化

**並列処理**:
- 現在は単一プロセス
- 将来: 複数ファイルの並列フィッティング

---

## 9. セキュリティ考慮事項

### 9.1 ファイルアップロード

- ファイルサイズ制限: Streamlitデフォルト（200MB）
- ファイル形式検証: 拡張子チェック
- コンテンツ検証: パース時のエラーハンドリング

### 9.2 データプライバシー

- ローカル実行を推奨
- アップロードされたデータはサーバー保存しない
- セッション終了時にデータクリア

---

## 10. 将来の拡張

### 10.1 短期的な改善
- [ ] セッション読み込み機能（JSONファイルからの復元）
- [ ] 解析範囲の選択（周波数範囲のスライダー）
- [ ] ファイルの並び替え（ドラッグ&ドロップ）
- [ ] データの部分非表示（特定の周波数範囲を除外）

### 10.2 中期的な改善
- [ ] 自動フィッティング機能の改善
- [ ] 複数回路モデルの比較
- [ ] Kramers-Kronig検証
- [ ] DRTデータの解析

### 10.3 長期的な改善
- [ ] 機械学習による回路モデル推定
- [ ] データベース連携
- [ ] マルチユーザー対応
- [ ] API提供

---

## 11. 参考資料

### 11.1 インピーダンス分光法
- Barsoukov, E., & Macdonald, J. R. (2018). *Impedance Spectroscopy: Theory, Experiment, and Applications*. Wiley.

### 11.2 等価回路フィッティング
- pyEIS: https://github.com/ECSHackWeek/impedance.py
- EIS Analyzer (Gamry): https://www.gamry.com/

### 11.3 技術ドキュメント
- Streamlit Documentation: https://docs.streamlit.io/
- Plotly Python: https://plotly.com/python/
- SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html

---

**Document End**
