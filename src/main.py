"""モジュール"""
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import math
import base64
import numpy as np
import itertools
import japanize_matplotlib # type: ignore  # noqa: F401
from typing import List, Dict, Any, Tuple, Union

class DataLoader:
   """データを指定されたファイルパスから読み込むクラス"""

   def __init__(self, data_path: str):
      """
      DataLoaderクラスの初期化メソッド。

      Args:
         data_path (str): 読み込むデータファイルのパス
      """
      self.data_path = data_path

   def load_data(self) -> pd.DataFrame:
      """
      指定されたパスからCSVファイルを読み込み、データをpandas DataFrameとして返す

      Returns:
         pd.DataFrame: 読み込まれたデータのDataFrame
         ファイルが見つからない場合は空のDataFrameを返す

      Raises:
         FileNotFoundError: ファイルが見つからなかった場合に例外をキャッチし、エラーメッセージを表示
      """
      try:
         return pd.read_csv(self.data_path)
      except FileNotFoundError as e:
         st.error("データファイルが見つかりません。ファイルパスを確認してください。")
         st.error(f"エラー内容: {str(e)}")
         return pd.DataFrame()

class PageSetup:
   """Streamlitアプリケーションのページ設定を行うクラス"""

   def __init__(self, css_file: str, img_file: str):
      """
      PageSetupクラスの初期化メソッド

      Args:
         css_file (str): 読み込むCSSファイルのパス
         img_file (str): 読み込むロゴ画像ファイルのパス
      """
      self.css_file = css_file
      self.img_file = img_file

   def setup(self) -> None:
      """
      ページの設定、CSSの読み込み、ロゴとタイトルの表示を行うメインメソッド
      """
      self._set_page_config()
      self._load_css()
      self._display_logo()
      self._display_title()

   def _set_page_config(self) -> None:
      """
      Streamlitページの基本設定を行う

      Raises:
         Exception: ページ設定に失敗した場合に例外をキャッチし、エラーメッセージを表示する
      """
      try:
         st.set_page_config(
               page_title="G1 Keiba Viewer",
               page_icon="🐎",
               layout="wide"
         )
      except Exception as e:
         st.error("ページ設定中にエラーが発生しました")
         st.error(f"エラー内容: {str(e)}")

   def _load_css(self) -> None:
      """
      指定されたCSSファイルを読み込み、ページに適用する

      Raises:
         Exception: CSSファイルの読み込みに失敗した場合に例外をキャッチし、エラーメッセージを表示する
      """
      try:
         with open(self.css_file, 'r', encoding='utf-8') as f:
               css = f.read()
         st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
      except Exception as e:
         st.error("CSSファイルの読み込み中にエラーが発生しました")
         st.error(f"エラー内容: {str(e)}")

   def _display_logo(self) -> None:
      """
      ロゴ画像を表示し、指定されたリンク先にリダイレクトするリンクを提供する

      Raises:
         Exception: 画像ファイルの読み込みに失敗した場合に例外をキャッチし、エラーメッセージを表示する
      """
      link_url = "https://www.jra.go.jp/datafile/seiseki/replay/2024/g1.html"
      try:
         with open(self.img_file, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
         st.markdown(f'''
            <a href="{link_url}" target="_blank" class="image-link">
                  <img src="data:image/gif;base64,{encoded_image}" alt="JRA Logo">
            </a>
         ''', unsafe_allow_html=True)
      except Exception as e:
         st.error(f"エラーが発生しました: {e}")

   def _display_title(self) -> None:
      """
      アプリケーションのタイトルとサブタイトルを表示する
      """
      st.markdown("# 🐎 G1 Keiba Viewer 🏁")
      st.markdown("##### 競馬の G1 レースデータを可視化するアプリ")

class DataVisualizer:
   """データの可視化を行うクラス"""

   def __init__(self, df: pd.DataFrame):
      """
      DataVisualizerクラスの初期化メソッド

      Args:
         df (pd.DataFrame): 可視化するデータのDataFrame
      """
      self.df = df

   def plot_order_distribution(self) -> None:
      """
      着順の人気分布を3つのヒストグラムとして描画する
      """
      st.subheader("着順の分布")
      col1, col2, col3 = st.columns(3)
      
      for i, (column, title) in enumerate([
         ('1着_人気', '1着_人気のヒストグラム'),
         ('2着_人気', '2着_人気のヒストグラム'),
         ('3着_人気', '3着_人気のヒストグラム')
      ]):
         with [col1, col2, col3][i]:
               fig, ax = plt.subplots()
               self._plot_histogram(self.df[column], title, ax, ['green', 'red', 'blue'][i])
               st.pyplot(fig)

   def _plot_histogram(self, column_data: pd.Series, title: str, ax: plt.Axes, color: str) -> None:
      """
      指定された列のヒストグラムを描画する

      Args:
         column_data (pd.Series): ヒストグラムを描画するデータ
         title (str): ヒストグラムのタイトル
         ax (plt.Axes): 描画対象のAxesオブジェクト
         color (str): ヒストグラムの色
      """
      bins = [i - 0.5 for i in range(1, 19)]
      sns.histplot(column_data, bins=bins, kde=True, ax=ax, color=color)
      ax.set_title(title)
      ax.set_xlim(0, 18)
      ax.set_ylim(0, 15)
      ax.set_xlabel('人気')
      ax.set_ylabel('回数')

   def plot_payouts_distribution(self) -> None:
      """
      払戻金額の分布(単勝, 馬連, 3連複)をヒストグラムとして描画する
      """
      st.subheader("払戻の分布(単勝, 馬連, 3連複)")
      columns = st.columns(3)
      
      histogram_data = [
         ('単勝_払戻', 'purple', '単勝払戻額'),
         ('馬連_払戻', 'orange', '馬連払戻額'),
         ('3連複_払戻', 'green', '3連複払戻額')
      ]
      
      for i, (column, color, title) in enumerate(histogram_data):
         with columns[i]:
               fig = self._create_histogram(self.df, column, color, title)
               st.pyplot(fig)

   def _create_histogram(self, data: pd.DataFrame, column: str, color: str, title: str) -> plt.Figure:
      """
      指定された列のヒストグラムを作成し、返す

      Args:
         data (pd.DataFrame): データのDataFrame
         column (str): ヒストグラムを描画する列名
         color (str): ヒストグラムの色
         title (str): ヒストグラムのタイトル

      Returns:
         plt.Figure: ヒストグラムの描画結果を含むFigureオブジェクト
      """
      fig, ax = plt.subplots()
      sns.histplot(data[column], bins=30, kde=True, color=color)
      plt.title(f'{title}のヒストグラム')
      ax.set_ylim(0, 25)
      ax.set_yticks(range(0, 25, 2))
      plt.xlabel('払戻額')
      plt.ylabel('回数')
      return fig


class BetSimulator:
   """競馬のベットシミュレーションを行うクラス"""

   def __init__(self, df: pd.DataFrame):
      """
      BetSimulatorクラスの初期化メソッド

      Args:
         df (pd.DataFrame): シミュレーションに使用するデータのDataFrame
      """
      self.df = df
      if 'bets' not in st.session_state:
         st.session_state['bets'] = []

   def add_bet(self, bet_type: str, horses: List[int], amount: int) -> None:
      """
      ベットを追加し、セッションに保存する

      Args:
         bet_type (str): ベットの種類（単勝, 馬連, など）
         horses (List[int]): ベットする馬のリスト
         amount (int): ベット金額
      """
      st.session_state['bets'].append({
         'type': bet_type,
         'horses': horses,
         'amount': amount
      })

   def display_bet_summary(self) -> None:
      """
      現在のベットの要約を表示し、総額を計算して表示する
      """
      total_amount = 0
      for bet_type, emoji in [("単勝", "🏇"), ("複勝", "🏇"), ("馬連", "🏇"), ("ワイド", "🏇"), ("馬単", "🏇"), ("3連複", "🏇"), ("3連単", "🏇")]:
         bet = next((b for b in reversed(st.session_state['bets']) if b['type'] == bet_type), None)
         if bet:
               horses = ', '.join(map(str, bet['horses']))
               amount = bet['amount']
               
               num_combinations = self._calculate_combinations(bet_type, len(bet['horses']))
               total = num_combinations * amount
               total_amount += total
               
               st.write(f"{emoji} 人気番号 {horses} に {amount} 円 {bet_type}ベットしているので、{total} 円 です")

      st.write(f"##### 💰️ 賭け金の全額は {total_amount} 円 です 💰️")

   def _calculate_combinations(self, bet_type: str, num_horses: int) -> int:
      """
      ベットタイプに応じて組み合わせの数を計算する

      Args:
         bet_type (str): ベットの種類
         num_horses (int): ベットする馬の数

      Returns:
         int: ベットに対する組み合わせの数
      """
      if bet_type in ["馬連", "ワイド"]:
         return math.comb(num_horses, 2)
      elif bet_type == "馬単":
         return math.perm(num_horses, 2)
      elif bet_type == "3連複":
         return math.comb(num_horses, 3)
      elif bet_type == "3連単":
         return math.perm(num_horses, 3)
      else:
         return num_horses

   def run_simulation(self, selected_bets: List[str]) -> Tuple[pd.DataFrame, List[float]]:
      """
      選択されたベットに基づいてシミュレーションを実行する

      Args:
         selected_bets (List[str]): シミュレーション対象のベットタイプのリスト

      Returns:
         Tuple[pd.DataFrame, List[float]]: シミュレーション結果を含むDataFrameと各ベットタイプの合計収支のリスト
      """
      total_profit_list = []
      for bet_type in selected_bets:
         eachyear_profit_list: List[float] = []
         eachyear_result_list: List[Union[float, str]] = []

         for _, row in self.df.iterrows():
               first, second, third = row['1着_人気'], row['2着_人気'], row['3着_人気']
               
               if pd.isna(row[f'{bet_type}_払戻']):
                  eachyear_profit_list.append(np.nan)
                  eachyear_result_list.append(np.nan)
                  continue

               payout_value = row[f'{bet_type}_払戻']
               bet_horses = set(st.session_state['bets'][-1]['horses'])
               amount = st.session_state['bets'][-1]['amount']

               num_combinations = self._calculate_combinations(bet_type, len(bet_horses))
               total_bet = num_combinations * amount
               
               payout, result = self._calculate_payout(bet_type, bet_horses, first, second, third, payout_value, total_bet)
               
               eachyear_profit_list.append(payout)
               eachyear_result_list.append(result)

         self.df[f'{bet_type}_的中'] = eachyear_result_list
         self.df[f'{bet_type}_収支'] = eachyear_profit_list
         
         total_profit = self.df[f'{bet_type}_収支'].sum(skipna=True)
         total_profit_list.append(total_profit)
         
         hit_rate = self.df[f'{bet_type}_的中'].value_counts(normalize=True, dropna=True) * 100
         if '⭕️' in hit_rate:
            st.write(f'{bet_type}_合計収支: {int(total_profit)}円, {bet_type}_的中率: {hit_rate["⭕️"]:.2f}%')
         else:
            st.write(f'{bet_type}_合計収支: {int(total_profit)}円, {bet_type}_的中率: 0.00%')
      
      return self.df, total_profit_list

   def _calculate_payout(self, bet_type: str, bet_horses: set, first: int, second: int, third: int, payout_value: float, total_bet: int) -> Tuple[int, str]:
      """
      ベットに対する払戻額を計算し、結果を返す

      Args:
         bet_type (str): ベットの種類
         bet_horses (set): ベットする馬のセット
         first (int): 1着の人気
         second (int): 2着の人気
         third (int): 3着の人気
         payout_value (float): 払戻額
         total_bet (int): 合計ベット金額

      Returns:
         Tuple[int, str]: 払戻金額と結果（的中 or 不的中）
      """
      if bet_type == "単勝":
         return (int(payout_value - total_bet), '⭕️') \
            if first in bet_horses else (-total_bet, '✕')
      elif bet_type == "複勝":
         return (int(payout_value - total_bet), '⭕️') \
            if any(horse in bet_horses for horse in [first, second, third]) else (-total_bet, '✕')
      elif bet_type == "馬連":
         return (int(payout_value - total_bet), '⭕️') \
            if first in bet_horses and second in bet_horses else (-total_bet, '✕')
      elif bet_type == "ワイド":
         return (int(payout_value - total_bet), '⭕️') \
            if len(bet_horses.intersection({first, second, third})) >= 2 else (-total_bet, '✕')
      elif bet_type == "馬単":
         return (int(payout_value - total_bet), '⭕️') \
            if len(bet_horses) >= 2 and first == list(bet_horses)[0] and second == list(bet_horses)[1] else (-total_bet, '✕')
      elif bet_type == "3連複":
         return (int(payout_value - total_bet), '⭕️') \
            if bet_horses.issuperset({first, second, third}) else (-total_bet, '✕')
      elif bet_type == "3連単":
         return (int(payout_value - total_bet), '⭕️') \
            if len(bet_horses) >= 3 and first == list(bet_horses)[0] and second == list(bet_horses)[1] and third == list(bet_horses)[2] else (-total_bet, '✕')
      else:
         return (0, '✕')

class BetOptimizer:
   """ベットの最適化を行うクラス"""

   def __init__(self, df: pd.DataFrame, bet_type: str, threshold: float, amount: int):
      """
      BetOptimizerの初期化。

      Args:
         df (pd.DataFrame): レースデータのDataFrame
         bet_type (str): ベットタイプ（現在は「3連複」のみ対応）
         threshold (float): 的中率の閾値
         amount (int): 1口あたりのベット金額
      """
      self.df = df
      self.bet_type = bet_type
      self.threshold = threshold
      self.amount = amount

   def optimize(self) -> List[Dict[str, Any]]:
      """
      ベットの最適化を行い、上位10個の結果を返す。

      Returns:
         List[Dict[str, Any]]: 最適化されたベット結果のリスト
      """
      number_range = range(1, 11)
      results = []

      for bet_size in range(3, 11):
         for bet_num in itertools.combinations(number_range, bet_size):
            bet_count = math.comb(len(bet_num), 3)
            bet_amount = bet_count * self.amount

            # 利益と結果を計算
            profit_list, result_list = self._calculate_profits(bet_num, bet_amount)

            # 的中率を計算
            hit_count = result_list.count('⭕️')
            total_bets = len([r for r in result_list if r in ['⭕️', '✕']])
            hit_rate = (hit_count / total_bets) * 100 if total_bets > 0 else 0

            # 的中率が閾値を超えている場合、結果を追加
            if hit_rate >= self.threshold:
               total_profit = np.nansum(profit_list)
               results.append({
                  'bet_num': bet_num,
                  'bet_size': bet_size,
                  'total_profit': total_profit,
                  'hit_rate': hit_rate
               })

      # 合計収支でソートし、上位10個を返す
      return sorted(results, key=lambda x: float(x['total_profit']) \
         if isinstance(x['total_profit'], (int, float, str)) else float('nan'), reverse=True)[:10]

   def _calculate_profits(self, bet_num: Tuple[int, ...], bet_amount: int) -> Tuple[List[float], List[Union[float, str]]]:
      """
      ベットに対する利益と結果を計算する。

      Args:
         bet_num (Tuple[int, ...]): 賭ける人気番号の組み合わせ
         bet_amount (int): ベット金額

      Returns:
         Tuple[List[float], List[Union[float, str]]]: 利益リストと結果リスト
      """
      profit_list: List[float] = []
      result_list: List[Union[float, str]] = []

      for _, row in self.df.iterrows():
         first, second, third = row['1着_人気'], row['2着_人気'], row['3着_人気']

         if pd.isna(row[f'{self.bet_type}_払戻']):
            profit_list.append(np.nan)
            result_list.append(np.nan)
            continue

         payout_value = row[f'{self.bet_type}_払戻'] * self.amount / 100

         # 的中している場合は払戻額を計算
         if {first, second, third}.issubset(bet_num):
               payout = int(payout_value - bet_amount)
               result = '⭕️'
         else:
               payout = -bet_amount
               result = '✕'

         profit_list.append(payout)
         result_list.append(result)

      return profit_list, result_list


def main() -> None:
   """
   アプリケーションのメイン処理
   データの読み込み、可視化、ベットシミュレーション、および最適化を行う
   """
   # データファイルや画像ファイルのパス設定
   src_dir = os.path.dirname(__file__)
   data_path = os.path.join(src_dir, '../data/g1_result_haraimodoshi.csv')
   css_file = os.path.join(src_dir, 'style.css')
   img_file = os.path.join(src_dir, '../img/keiba.gif')

   # ページセットアップ
   page_setup = PageSetup(css_file, img_file)
   page_setup.setup()

   # データの読み込み
   data_loader = DataLoader(data_path)
   org_df = data_loader.load_data()

   # レースの辞書を定義
   race_dict = {
      "": "選択してください",
      "feb": "フェブラリーステークス",
      "taka": "高松宮記念",
      "osaka": "大阪杯",
      "ouka": "桜花賞",
      "satuki": "皐月賞",
      "haruten": "天皇賞（春）",
      "nhk": "NHKマイルカップ",
      "victoria": "ヴィクトリアマイル",
      "oaks": "オークス",
      "derby": "日本ダービー",
      "yasuda": "安田記念",
      "takara": "宝塚記念",
      "sprinter": "スプリンターズステークス",
      "shuka": "秋華賞",
      "kikka": "菊花賞",
      "akiten": "天皇賞（秋）",
      "eliza": "エリザベス女王杯",
      "milec": "マイルチャンピオンシップ",
      "japanc": "ジャパンカップ",
      "champc": "チャンピオンズカップ",
      "hanshinjf": "阪神ジュベナイルフィリーズ",
      "asahifs": "朝日杯フューチュリティステークス",
      "arima": "有馬記念",
      "hopeful": "ホープフルステークス"
   }

   # レース選択セクション
   st.subheader("🏁 レースの選択")
   racename = st.selectbox("レース名(時系列順)", list(race_dict.values()))

   if racename == "選択してください":
      st.info("ℹ️ レース名を選択してください")
      return

   # 選択されたレースのデータをフィルタリング
   racename_en = next(k for k, v in race_dict.items() if v == racename)
   selected_df = org_df[org_df["レース名"] == racename_en]
   min_year, max_year = selected_df['年'].min(), selected_df['年'].max()

   st.write(f"###### 選択中のレースデータ({min_year}年-{max_year}年)")
   st.dataframe(selected_df)

   # データ可視化セクション
   # 2つのカラムを作成
   col1, col2 = st.columns(2)
   # 左のカラムに着順のチェックボックス
   with col1:
      show_order = st.checkbox("🏆️ 着順の分布を見る")
   # 右のカラムに払戻のチェックボックス
   with col2:
      show_payouts = st.checkbox("💰️ 払戻の分布を見る")

   if not show_order and not show_payouts:
      st.info("ℹ️ 可視化するデータを選択してください")
   
   visualizer = DataVisualizer(selected_df)
   if show_order:
      visualizer.plot_order_distribution()
   if show_payouts:
      visualizer.plot_payouts_distribution()

   # ベットシミュレーションセクション
   st.subheader("♠️ ベットシミュレーション")
   bet_types = ["単勝", "複勝", "馬連", "ワイド", "馬単", "3連複", "3連単"]
   cols = st.columns(len(bet_types))
   selected_bets = [bet_type for i, bet_type in enumerate(bet_types) if cols[i].toggle(f"{bet_type}", value=False)]

   if selected_bets:
      st.write(f"選択されたベットタイプ: {', '.join(selected_bets)}")
      simulator = BetSimulator(selected_df)

      # ベットの入力と追加
      for bet_type in selected_bets:
         st.write(f"🏇 {bet_type}")
         horses = st.multiselect(f"{bet_type}のベットする馬の人気番号を選択してください", list(range(1, 19)), key=f"{bet_type}_bet_horse")
         amount = st.number_input(f"{bet_type}のベット金額を入力してください", min_value=100, max_value=100000, step=100, value=100, key=f"{bet_type}_bet_amount")
         if st.button(f"{bet_type}ベットを追加", key=f"{bet_type}_button"):
            simulator.add_bet(bet_type, horses, int(amount))

      simulator.display_bet_summary()

      if st.button("シミュレーションを実行"):
         result_df, total_profit_list = simulator.run_simulation(selected_bets)
         st.dataframe(result_df)
         final_profit = sum(total_profit_list)
         st.write(f'##### 全体収支: {int(final_profit)}円')
   else:
      st.warning("⚠️ ベットタイプを選択してください")

   # 最適化セクション
   st.subheader("⚖️ 最適化(現在3連複のみ実装済)")
   threshold = st.slider("的中率の閾値を設定してください", min_value=0, max_value=100, value=50)
   amount = st.number_input("1口あたりのベット金額を入力してください", min_value=100, max_value=100000, step=100, value=100)
   
   if st.button("最適化を実行"):
      optimizer = BetOptimizer(selected_df, "3連複", threshold, int(amount))
      top_results = optimizer.optimize()
      
      if top_results:
         top_df = pd.DataFrame({
            '順位': list(range(1, len(top_results) + 1)),
            '賭ける人気番号のセット': [result['bet_num'] for result in top_results],
            '選択頭数': [result['bet_size'] for result in top_results],
            '合計収支': [result['total_profit'] for result in top_results],
            '的中率': [f"{result['hit_rate']:.2f}%" for result in top_results]
         })
         st.dataframe(top_df)
      else:
         st.warning("指定された条件を満たす結果が見つかりませんでした。閾値を下げてみてください。")

   st.markdown("#### All rights reserved. ©️ 2024 Yusei Ito")

if __name__ == "__main__":
   main()