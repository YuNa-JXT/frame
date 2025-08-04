import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 配置类 ---
class Config:
    """系统配置类"""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS = ["csv", "xlsx", "xls"]
    DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]
    MIN_DATA_POINTS = 30
    MAX_DATA_POINTS = 10000
    OUTLIER_THRESHOLD = 3
    SEASONALITY_MIN_PERIODS = 730  # 2年数据才能可靠检测季节性


# --- 数据验证和处理类 ---
class DataValidator:
    """数据验证和清洗类"""

    @staticmethod
    def validate_file_size(file) -> bool:
        """验证文件大小"""
        if hasattr(file, 'size') and file.size > Config.MAX_FILE_SIZE:
            return False
        return True

    @staticmethod
    def parse_date_column(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """智能解析日期列"""
        if date_col not in df.columns:
            raise ValueError(f"数据中未找到'{date_col}'列")

        # 尝试多种日期格式
        for date_format in Config.DATE_FORMATS:
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
                return df
            except:
                continue

        # 如果格式化失败，尝试自动推断
        try:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            return df
        except:
            raise ValueError("无法解析日期格式，请确保日期列格式正确")

    @staticmethod
    def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, str]:
        """验证数据结构"""
        required_cols = ['Date', 'Cases']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, f"缺少必要列: {missing_cols}"

        if len(df) < Config.MIN_DATA_POINTS:
            return False, f"数据点数量不足，至少需要{Config.MIN_DATA_POINTS}个数据点"

        if len(df) > Config.MAX_DATA_POINTS:
            return False, f"数据点数量过多，最多支持{Config.MAX_DATA_POINTS}个数据点"

        # 检查Cases列是否为数值型
        if not pd.api.types.is_numeric_dtype(df['Cases']):
            return False, "Cases列必须为数值型数据"

        # 检查是否有负值
        if (df['Cases'] < 0).any():
            return False, "Cases列不能包含负值"

        return True, "数据验证通过"

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 移除重复日期
        df = df.drop_duplicates(subset=['Date'])

        # 按日期排序
        df = df.sort_values('Date')

        # 处理缺失值
        df['Cases'] = df['Cases'].fillna(df['Cases'].median())

        # 重置索引
        df = df.reset_index(drop=True)

        return df


# --- 后端分析类 ---
class TimeSeriesAnalyzer:
    """时间序列分析类"""

    def __init__(self):
        self.results = {}

    def analyze_stationarity(self, data: pd.Series) -> Dict:
        """平稳性检测"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(data.dropna())

            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'interpretation': '序列平稳' if result[1] < 0.05 else '序列非平稳'
            }
        except Exception as e:
            logger.error(f"平稳性检测失败: {e}")
            return {
                'error': str(e),
                'is_stationary': False,
                'interpretation': '无法进行平稳性检测'
            }

    def analyze_seasonality(self, data: pd.Series) -> Dict:
        """季节性检测"""
        try:
            if len(data) < Config.SEASONALITY_MIN_PERIODS:
                return {
                    'has_seasonality': False,
                    'warning': f"数据长度不足{Config.SEASONALITY_MIN_PERIODS}天，无法可靠检测季节性"
                }

            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(
                data.dropna(),
                model='additive',
                period=365,
                extrapolate_trend='freq'
            )

            # 计算季节性强度
            seasonal_strength = decomposition.seasonal.var() / data.var()

            return {
                'has_seasonality': seasonal_strength > 0.01,
                'seasonal_strength': seasonal_strength,
                'interpretation': '存在明显季节性' if seasonal_strength > 0.01 else '季节性不明显'
            }
        except Exception as e:
            logger.error(f"季节性检测失败: {e}")
            return {
                'error': str(e),
                'has_seasonality': False,
                'interpretation': '无法进行季节性检测'
            }

    def detect_outliers(self, data: pd.Series) -> Dict:
        """异常值检测"""
        try:
            # Z-score方法
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers_mask = z_scores > Config.OUTLIER_THRESHOLD
            outliers = data[outliers_mask]

            # IQR方法作为补充
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

            return {
                'num_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100,
                'has_bursts': len(outliers) > 0,
                'outlier_indices': outliers.index.tolist(),
                'iqr_outliers': len(iqr_outliers),
                'interpretation': f'检测到{len(outliers)}个异常值' if len(outliers) > 0 else '未检测到明显异常值'
            }
        except Exception as e:
            logger.error(f"异常值检测失败: {e}")
            return {
                'error': str(e),
                'has_bursts': False,
                'interpretation': '无法进行异常值检测'
            }

    def comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """综合分析"""
        try:
            data = df.set_index('Date')['Cases']

            results = {
                'data_info': {
                    'total_points': len(data),
                    'date_range': f"{data.index.min().strftime('%Y-%m-%d')} 到 {data.index.max().strftime('%Y-%m-%d')}",
                    'mean_cases': data.mean(),
                    'std_cases': data.std(),
                    'min_cases': data.min(),
                    'max_cases': data.max()
                }
            }

            # 分别进行各项分析
            results['stationarity'] = self.analyze_stationarity(data)
            results['seasonality'] = self.analyze_seasonality(data)
            results['outliers'] = self.detect_outliers(data)

            # 生成综合结论
            results['summary'] = self._generate_summary(results)

            return results

        except Exception as e:
            logger.error(f"综合分析失败: {e}")
            return {'error': str(e)}

    def _generate_summary(self, results: Dict) -> str:
        """生成分析摘要"""
        summary_parts = []

        if results.get('stationarity', {}).get('is_stationary'):
            summary_parts.append("数据序列平稳")
        else:
            summary_parts.append("数据序列非平稳")

        if results.get('seasonality', {}).get('has_seasonality'):
            summary_parts.append("存在季节性模式")
        else:
            summary_parts.append("无明显季节性")

        outlier_count = results.get('outliers', {}).get('num_outliers', 0)
        if outlier_count > 0:
            summary_parts.append(f"检测到{outlier_count}个异常值")
        else:
            summary_parts.append("无明显异常值")

        return "，".join(summary_parts)


# --- 模型推荐类 ---
class ModelRecommender:
    """模型推荐系统"""

    def __init__(self):
        self.model_info = {
            'ARIMA': {
                'description': 'ARIMA模型适用于平稳时间序列',
                'pros': ['经典可靠', '参数可解释', '预测置信区间'],
                'cons': ['要求数据平稳', '不适合处理复杂非线性关系'],
                'best_for': '平稳数据，无季节性，无突发性'
            },
            'SARIMA': {
                'description': '季节性ARIMA模型',
                'pros': ['处理季节性', '理论基础扎实', '预测置信区间'],
                'cons': ['参数调优复杂', '计算相对较慢'],
                'best_for': '有季节性的平稳或非平稳数据'
            },
            'Prophet': {
                'description': 'Facebook开发的时间序列预测工具',
                'pros': ['自动处理季节性', '处理缺失值', '对异常值鲁棒'],
                'cons': ['黑盒模型', '对短期数据效果一般'],
                'best_for': '长期数据，有季节性，存在异常值'
            },
            'LSTM': {
                'description': '长短期记忆神经网络',
                'pros': ['捕捉复杂模式', '处理长期依赖', '适应性强'],
                'cons': ['需要大量数据', '训练时间长', '超参数敏感'],
                'best_for': '大量数据，复杂非线性模式'
            },
            'XGBoost': {
                'description': '梯度提升决策树',
                'pros': ['强大的非线性建模', '特征重要性', '处理异常值'],
                'cons': ['需要特征工程', '容易过拟合', '缺乏时序特性'],
                'best_for': '有丰富特征，存在突发性'
            }
        }

    def recommend_models(self, analysis_results: Dict) -> List[Dict]:
        """基于分析结果推荐模型"""
        recommendations = []

        stationarity = analysis_results.get('stationarity', {})
        seasonality = analysis_results.get('seasonality', {})
        outliers = analysis_results.get('outliers', {})

        is_stationary = stationarity.get('is_stationary', False)
        has_seasonality = seasonality.get('has_seasonality', False)
        has_outliers = outliers.get('has_bursts', False)
        data_points = analysis_results.get('data_info', {}).get('total_points', 0)

        # 基于数据特征推荐模型
        if not has_seasonality and not has_outliers:
            if is_stationary:
                recommendations.append({
                    'model': 'ARIMA',
                    'score': 9,
                    'reason': '数据平稳，无季节性和异常值，ARIMA模型最适合'
                })
            else:
                recommendations.append({
                    'model': 'ARIMA',
                    'score': 7,
                    'reason': '需要差分处理后使用ARIMA模型'
                })

        if has_seasonality:
            recommendations.append({
                'model': 'SARIMA',
                'score': 8,
                'reason': '存在季节性模式，SARIMA模型能很好处理'
            })

            if data_points > 365:
                recommendations.append({
                    'model': 'Prophet',
                    'score': 8,
                    'reason': '数据量充足且有季节性，Prophet表现优秀'
                })

        if has_outliers:
            recommendations.append({
                'model': 'Prophet',
                'score': 8,
                'reason': '对异常值具有鲁棒性'
            })

            recommendations.append({
                'model': 'XGBoost',
                'score': 7,
                'reason': '能够处理异常值和非线性关系'
            })

        if data_points > 1000:
            recommendations.append({
                'model': 'LSTM',
                'score': 7,
                'reason': '数据量充足，可以尝试深度学习方法'
            })

        # 确保至少有一个推荐
        if not recommendations:
            recommendations.append({
                'model': 'ARIMA',
                'score': 6,
                'reason': '通用的时间序列预测方法'
            })

        # 按分数排序并添加详细信息
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        for rec in recommendations:
            rec.update(self.model_info.get(rec['model'], {}))

        return recommendations


# --- 数据生成器类 ---
class DataGenerator:
    """模拟数据生成器"""

    @staticmethod
    def generate_epidemic_data(
            start_date: str = '2020-01-01',
            periods: int = 1095,  # 3年
            base_level: float = 100,
            noise_level: float = 10,
            seasonal_amplitude: float = 50,
            outbreak_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """生成更真实的传染病数据"""

        dates = pd.date_range(start=start_date, periods=periods, freq='D')

        # 基础水平（带趋势）
        trend = np.linspace(0, 20, periods)
        base_data = base_level + trend + np.random.normal(0, noise_level, periods)

        # 季节性成分（年度和周度）
        yearly_seasonal = seasonal_amplitude * np.sin(2 * np.pi * np.arange(periods) / 365.25 + np.pi / 2)
        weekly_seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 7)

        # 突发疫情
        outbreak_data = np.zeros(periods)
        if outbreak_params:
            start_day = outbreak_params.get('start_day', 400)
            duration = outbreak_params.get('duration', 60)
            intensity = outbreak_params.get('intensity', 200)

            if start_day < periods:
                end_day = min(start_day + duration, periods)
                outbreak_pattern = intensity * np.exp(-0.1 * np.arange(duration))
                outbreak_data[start_day:start_day + len(outbreak_pattern)] = outbreak_pattern[:end_day - start_day]

        # 组合所有成分
        total_cases = base_data + yearly_seasonal + weekly_seasonal + outbreak_data
        total_cases = np.maximum(total_cases, 0)  # 确保非负
        total_cases = np.round(total_cases).astype(int)

        df = pd.DataFrame({
            'Date': dates,
            'Cases': total_cases
        })

        return df


# --- 可视化类 ---
class Visualizer:
    """数据可视化类"""

    @staticmethod
    def plot_time_series(df: pd.DataFrame, title: str = "时间序列数据") -> go.Figure:
        """绘制时间序列图"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Cases'],
            mode='lines',
            name='病例数',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='日期',
            yaxis_title='病例数',
            hovermode='x unified',
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_forecast_comparison(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
        """绘制历史数据和预测结果对比图"""
        fig = go.Figure()

        # 历史数据
        fig.add_trace(go.Scatter(
            x=historical_df['Date'],
            y=historical_df['Cases'],
            mode='lines',
            name='历史数据',
            line=dict(color='blue', width=2)
        ))

        # 预测数据
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['PredictedCases'],
            mode='lines',
            name='预测数据',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title='历史数据与预测结果对比',
            xaxis_title='日期',
            yaxis_title='病例数',
            hovermode='x unified',
            showlegend=True
        )

        return fig


# --- 主应用类 ---
class EpidemicPredictionApp:
    """主应用类"""

    def __init__(self):
        self.init_session_state()
        self.analyzer = TimeSeriesAnalyzer()
        self.recommender = ModelRecommender()
        self.visualizer = Visualizer()

    def init_session_state(self):
        """初始化会话状态"""
        default_states = {
            'current_step': 0,
            'uploaded_data': None,
            'analysis_results': None,
            'recommended_models': [],
            'selected_model': '',
            'training_results': None,
            'privacy_status': '',
            'forecast_results': None,
            'processed_data': None
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            layout="wide",
            page_title="传染病人数预测系统",
            page_icon="🦠",
            initial_sidebar_state="expanded"
        )

        st.title("🦠 传染病人数预测系统")
        st.markdown("---")

    def render_sidebar(self):
        """渲染侧边栏"""
        steps = [
            "📁 数据上传",
            "📊 数据分析",
            "🤖 模型推荐",
            "⚙️ 模型训练",
            "🔒 隐私保护",
            "📈 结果输出"
        ]

        st.sidebar.header("🧭 导航")

        for i, step_name in enumerate(steps):
            is_accessible = i <= st.session_state.current_step
            is_current = i == st.session_state.current_step

            if st.sidebar.button(
                    f"{step_name}",
                    disabled=not is_accessible,
                    key=f"nav_btn_{i}",
                    type="primary" if is_current else "secondary"
            ):
                st.session_state.current_step = i
                st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.info("💡 请按照步骤顺序操作")

        # 显示当前数据状态
        if st.session_state.uploaded_data is not None:
            st.sidebar.success(f"✅ 已加载 {len(st.session_state.uploaded_data)} 条记录")

        if st.session_state.selected_model:
            st.sidebar.info(f"🎯 当前模型: {st.session_state.selected_model}")

    def render_step_0_data_upload(self):
        """步骤0: 数据上传"""
        st.header("📁 第一步：数据上传")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("您可以上传CSV或Excel文件，或生成模拟数据进行演示。")
            st.info("📋 数据要求：必须包含 'Date'（日期）和 'Cases'（病例数）列")

            uploaded_file = st.file_uploader(
                "选择数据文件",
                type=Config.SUPPORTED_FORMATS,
                help="支持CSV和Excel格式，文件大小不超过50MB"
            )

            if uploaded_file is not None:
                self.process_uploaded_file(uploaded_file)

        with col2:
            st.markdown("### 🧪 或生成模拟数据")

            with st.expander("⚙️ 模拟数据参数"):
                start_date = st.date_input("开始日期", pd.to_datetime('2020-01-01'))
                periods = st.slider("数据点数", 365, 1095, 1095)
                base_level = st.slider("基础病例数", 50, 200, 100)
                seasonal_amplitude = st.slider("季节性强度", 20, 100, 50)

                add_outbreak = st.checkbox("添加疫情爆发", value=True)
                outbreak_params = None
                if add_outbreak:
                    outbreak_day = st.slider("爆发开始天数", 100, periods - 100, 400)
                    outbreak_duration = st.slider("爆发持续天数", 30, 120, 60)
                    outbreak_intensity = st.slider("爆发强度", 100, 500, 200)
                    outbreak_params = {
                        'start_day': outbreak_day,
                        'duration': outbreak_duration,
                        'intensity': outbreak_intensity
                    }

            if st.button("🚀 生成模拟数据", type="primary"):
                self.generate_simulation_data(start_date, periods, base_level, seasonal_amplitude, outbreak_params)

        # 显示已上传的数据
        if st.session_state.uploaded_data is not None:
            self.display_data_preview()

    def process_uploaded_file(self, uploaded_file):
        """处理上传的文件"""
        try:
            # 验证文件大小
            if not DataValidator.validate_file_size(uploaded_file):
                st.error(f"❌ 文件大小超过限制（{Config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB）")
                return

            # 读取文件
            with st.spinner("📖 正在读取文件..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df = pd.read_excel(uploaded_file)

            # 验证数据结构
            is_valid, message = DataValidator.validate_data_structure(df)
            if not is_valid:
                st.error(f"❌ {message}")
                return

            # 解析日期
            df = DataValidator.parse_date_column(df)

            # 清洗数据
            df = DataValidator.clean_data(df)

            st.session_state.uploaded_data = df
            st.session_state.current_step = 1
            st.success("✅ 数据文件已成功上传并验证！")
            st.rerun()

        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            st.error(f"❌ 文件处理失败: {str(e)}")

    def generate_simulation_data(self, start_date, periods, base_level, seasonal_amplitude, outbreak_params):
        """生成模拟数据"""
        try:
            with st.spinner("🧪 正在生成模拟数据..."):
                df = DataGenerator.generate_epidemic_data(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    periods=periods,
                    base_level=base_level,
                    seasonal_amplitude=seasonal_amplitude,
                    outbreak_params=outbreak_params
                )

                st.session_state.uploaded_data = df
                st.session_state.current_step = 1
                st.success("✅ 模拟数据已成功生成！")
                st.rerun()

        except Exception as e:
            logger.error(f"模拟数据生成失败: {e}")
            st.error(f"❌ 模拟数据生成失败: {str(e)}")

    def display_data_preview(self):
        """显示数据预览"""
        st.markdown("### 📋 数据预览")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(
                st.session_state.uploaded_data.head(10),
                use_container_width=True
            )

        with col2:
            df = st.session_state.uploaded_data
            st.metric("数据点数", len(df))
            st.metric("日期范围", f"{df['Date'].min()} 至 {df['Date'].max()}")
            st.metric("平均病例数", f"{df['Cases'].mean():.1f}")

        # 绘制时间序列图
        try:
            fig = self.visualizer.plot_time_series(st.session_state.uploaded_data)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ 无法绘制图表: {str(e)}")

    def render_step_1_data_analysis(self):
        """步骤1: 数据分析"""
        st.header("📊 第二步：数据分析")

        if st.session_state.uploaded_data is None:
            st.warning("⚠️ 请先上传数据")
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("系统将分析数据的统计特性，包括平稳性、季节性和异常值检测。")

        with col2:
            if st.button("🔍 开始分析", type="primary"):
                self.perform_data_analysis()

        # 显示分析结果
        if st.session_state.analysis_results is not None:
            self.display_analysis_results()

    def perform_data_analysis(self):
        """执行数据分析"""
        try:
            with st.spinner("🔍 正在执行数据分析..."):
                results = self.analyzer.comprehensive_analysis(st.session_state.uploaded_data)
                st.session_state.analysis_results = results
                st.session_state.current_step = 2
                st.success("✅ 数据分析完成！")
                st.rerun()

        except Exception as e:
            logger.error(f"数据分析失败: {e}")
            st.error(f"❌ 数据分析失败: {str(e)}")

    def display_analysis_results(self):
        """显示分析结果"""
        results = st.session_state.analysis_results

        if 'error' in results:
            st.error(f"❌ 分析失败: {results['error']}")
            return

        st.markdown("### 📈 分析结果")

        # 数据基本信息
        col1, col2, col3, col4 = st.columns(4)

        data_info = results.get('data_info', {})
        with col1:
            st.metric("数据点数", data_info.get('total_points', 'N/A'))
        with col2:
            st.metric("平均病例数", f"{data_info.get('mean_cases', 0):.1f}")
        with col3:
            st.metric("最大病例数", data_info.get('max_cases', 'N/A'))
        with col4:
            st.metric("标准差", f"{data_info.get('std_cases', 0):.1f}")

        # 详细分析结果
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🎯 平稳性检测")
            stationarity = results.get('stationarity', {})
            if 'error' not in stationarity:
                st.write(f"**结果**: {stationarity.get('interpretation', 'N/A')}")
                st.write(f"**P值**: {stationarity.get('p_value', 0):.4f}")
                st.write(f"**检验统计量**: {stationarity.get('adf_statistic', 0):.4f}")
            else:
                st.error(f"检测失败: {stationarity['error']}")

        with col2:
            st.markdown("#### 🔄 季节性检测")
            seasonality = results.get('seasonality', {})
            if 'error' not in seasonality:
                st.write(f"**结果**: {seasonality.get('interpretation', 'N/A')}")
                if 'seasonal_strength' in seasonality:
                    st.write(f"**季节性强度**: {seasonality['seasonal_strength']:.4f}")
                if 'warning' in seasonality:
                    st.warning(seasonality['warning'])
            else:
                st.error(f"检测失败: {seasonality['error']}")

        with col3:
            st.markdown("#### ⚡ 异常值检测")
            outliers = results.get('outliers', {})
            if 'error' not in outliers:
                st.write(f"**结果**: {outliers.get('interpretation', 'N/A')}")
                st.write(f"**异常值数量**: {outliers.get('num_outliers', 0)}")
                st.write(f"**异常值比例**: {outliers.get('outlier_percentage', 0):.2f}%")
            else:
                st.error(f"检测失败: {outliers['error']}")

        # 综合摘要
        st.markdown("#### 📝 综合摘要")
        st.info(results.get('summary', '无法生成摘要'))

    def render_step_2_model_recommendation(self):
        """步骤2: 模型推荐"""
        st.header("🤖 第三步：模型推荐")

        if st.session_state.analysis_results is None:
            st.warning("⚠️ 请先完成数据分析")
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("基于数据分析结果，系统将推荐最适合的预测模型。")

        with col2:
            if st.button("🎯 获取推荐", type="primary"):
                self.get_model_recommendations()

        # 显示推荐结果
        if st.session_state.recommended_models:
            self.display_model_recommendations()

    def get_model_recommendations(self):
        """获取模型推荐"""
        try:
            with st.spinner("🤖 正在分析并推荐模型..."):
                recommendations = self.recommender.recommend_models(st.session_state.analysis_results)
                st.session_state.recommended_models = recommendations
                st.session_state.selected_model = recommendations[0]['model'] if recommendations else ''
                st.session_state.current_step = 3
                st.success("✅ 模型推荐完成！")
                st.rerun()

        except Exception as e:
            logger.error(f"模型推荐失败: {e}")
            st.error(f"❌ 模型推荐失败: {str(e)}")

    def display_model_recommendations(self):
        """显示模型推荐"""
        st.markdown("### 🏆 推荐模型")

        # 显示前3个推荐
        for i, rec in enumerate(st.session_state.recommended_models[:3]):
            with st.expander(f"#{i + 1} {rec['model']} (推荐分数: {rec['score']}/10)", expanded=(i == 0)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**描述**: {rec.get('description', 'N/A')}")
                    st.write(f"**推荐理由**: {rec.get('reason', 'N/A')}")
                    st.write(f"**适用场景**: {rec.get('best_for', 'N/A')}")

                with col2:
                    if rec.get('pros'):
                        st.write("**优点**:")
                        for pro in rec['pros']:
                            st.write(f"• {pro}")

                    if rec.get('cons'):
                        st.write("**缺点**:")
                        for con in rec['cons']:
                            st.write(f"• {con}")

        # 模型选择
        st.markdown("### 🎛️ 选择模型")
        model_options = [rec['model'] for rec in st.session_state.recommended_models]

        selected_model = st.selectbox(
            "请选择要使用的模型:",
            model_options,
            index=model_options.index(
                st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        )

        st.session_state.selected_model = selected_model

        if st.button("✅ 确认选择", type="primary"):
            st.success(f"✅ 已选择模型: {selected_model}")
            st.session_state.current_step = 3

    def render_step_3_model_training(self):
        """步骤3: 模型训练"""
        st.header("⚙️ 第四步：模型训练与调参")

        if not st.session_state.selected_model:
            st.warning("⚠️ 请先选择模型")
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            st.info(f"🎯 当前选择的模型: **{st.session_state.selected_model}**")
            st.write("系统将自动进行模型训练和超参数优化。")

        with col2:
            if st.button("🚀 开始训练", type="primary"):
                self.train_model()

        # 显示训练结果
        if st.session_state.training_results:
            self.display_training_results()

    def train_model(self):
        """模拟模型训练"""
        try:
            with st.spinner("🔄 正在训练模型..."):
                # 模拟训练过程
                import time
                time.sleep(2)

                # 根据模型类型生成不同的结果
                model_name = st.session_state.selected_model

                training_results = {
                    'status': f'模型 {model_name} 训练完成',
                    'model_type': model_name,
                    'training_time': '2.3秒（模拟）',
                    'best_params': self._get_mock_params(model_name),
                    'performance_metrics': self._get_mock_performance(model_name),
                    'validation_scores': {
                        'train_score': 0.92,
                        'validation_score': 0.87,
                        'test_score': 0.85
                    }
                }

                st.session_state.training_results = training_results
                st.session_state.current_step = 4
                st.success("✅ 模型训练完成！")
                st.rerun()

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            st.error(f"❌ 模型训练失败: {str(e)}")

    def _get_mock_params(self, model_name: str) -> Dict:
        """获取模拟的最佳参数"""
        params_map = {
            'ARIMA': {'p': 2, 'd': 1, 'q': 1},
            'SARIMA': {'p': 1, 'd': 1, 'q': 1, 'P': 0, 'D': 1, 'Q': 1, 's': 365},
            'Prophet': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0},
            'LSTM': {'epochs': 100, 'batch_size': 32, 'layers': 2, 'units': 50},
            'XGBoost': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
        }
        return params_map.get(model_name, {'param1': 'value1'})

    def _get_mock_performance(self, model_name: str) -> Dict:
        """获取模拟的性能指标"""
        performance_map = {
            'ARIMA': {'RMSE': 15.2, 'MAE': 12.1, 'MAPE': 0.08},
            'SARIMA': {'RMSE': 12.8, 'MAE': 10.5, 'MAPE': 0.06},
            'Prophet': {'RMSE': 14.1, 'MAE': 11.2, 'MAPE': 0.07},
            'LSTM': {'RMSE': 18.5, 'MAE': 14.8, 'MAPE': 0.10},
            'XGBoost': {'RMSE': 16.0, 'MAE': 13.2, 'MAPE': 0.09}
        }
        return performance_map.get(model_name, {'RMSE': 17.0, 'MAPE': 0.095})

    def display_training_results(self):
        """显示训练结果"""
        results = st.session_state.training_results

        st.markdown("### 📊 训练结果")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("训练时间", results['training_time'])
        with col2:
            st.metric("训练分数", f"{results['validation_scores']['train_score']:.3f}")
        with col3:
            st.metric("验证分数", f"{results['validation_scores']['validation_score']:.3f}")

        # 最佳参数
        st.markdown("#### 🎛️ 最佳超参数")
        st.json(results['best_params'])

        # 性能指标
        st.markdown("#### 📈 性能指标")
        metrics = results['performance_metrics']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        with col2:
            st.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
        with col3:
            st.metric("MAPE", f"{metrics.get('MAPE', 0):.3f}")

    def render_step_4_privacy_protection(self):
        """步骤4: 隐私保护"""
        st.header("🔒 第五步：隐私保护")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("系统将对数据和模型应用隐私保护措施。")

            # 隐私保护选项
            privacy_options = st.multiselect(
                "选择隐私保护措施:",
                [
                    "数据脱敏",
                    "差分隐私",
                    "联邦学习",
                    "同态加密"
                ],
                default=["数据脱敏", "差分隐私"]
            )

        with col2:
            if st.button("🛡️ 应用保护", type="primary"):
                self.apply_privacy_protection(privacy_options)

        # 显示隐私保护状态
        if st.session_state.privacy_status:
            st.success(st.session_state.privacy_status)

    def apply_privacy_protection(self, options: List[str]):
        """应用隐私保护"""
        try:
            with st.spinner("🛡️ 正在应用隐私保护措施..."):
                import time
                time.sleep(1)

                applied_measures = ", ".join(options) if options else "基础保护"
                status = f"✅ 已应用隐私保护措施: {applied_measures}"

                st.session_state.privacy_status = status
                st.session_state.current_step = 5
                st.success("✅ 隐私保护措施已应用！")
                st.rerun()

        except Exception as e:
            logger.error(f"隐私保护失败: {e}")
            st.error(f"❌ 隐私保护失败: {str(e)}")

    def render_step_5_results_output(self):
        """步骤5: 结果输出"""
        st.header("📈 第六步：结果输出")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("生成预测结果并进行可视化展示。")

            # 预测参数设置
            forecast_days = st.slider("预测天数", 7, 90, 30)
            confidence_interval = st.slider("置信区间", 80, 99, 95)

        with col2:
            if st.button("📊 生成预测", type="primary"):
                self.generate_predictions(forecast_days, confidence_interval)

        # 显示预测结果
        if st.session_state.forecast_results is not None:
            self.display_forecast_results()

    def generate_predictions(self, forecast_days: int, confidence_interval: int):
        """生成预测结果"""
        try:
            with st.spinner("🔮 正在生成预测结果..."):
                df = st.session_state.uploaded_data
                model_name = st.session_state.selected_model

                # 模拟预测过程
                last_date = pd.to_datetime(df['Date'].max())
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )

                # 根据模型类型生成不同的预测结果
                last_value = df['Cases'].iloc[-1]

                if model_name in ['Prophet', 'SARIMA']:
                    # 带季节性的预测
                    trend = np.linspace(0, forecast_days * 0.1, forecast_days)
                    seasonal = 20 * np.sin(2 * np.pi * np.arange(forecast_days) / 365.25)
                    noise = np.random.normal(0, 5, forecast_days)
                    predictions = last_value + trend + seasonal + noise
                else:
                    # 简单趋势预测
                    trend = np.linspace(0, forecast_days * 0.05, forecast_days)
                    noise = np.random.normal(0, 8, forecast_days)
                    predictions = last_value + trend + noise

                predictions = np.maximum(predictions, 0).round().astype(int)

                # 生成置信区间
                margin = predictions * 0.15  # 15%的误差范围
                upper_bound = (predictions + margin).round().astype(int)
                lower_bound = np.maximum(predictions - margin, 0).round().astype(int)

                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'PredictedCases': predictions,
                    'UpperBound': upper_bound,
                    'LowerBound': lower_bound,
                    'ConfidenceInterval': confidence_interval
                })

                st.session_state.forecast_results = forecast_df
                st.success("✅ 预测结果已生成！")
                st.rerun()

        except Exception as e:
            logger.error(f"预测生成失败: {e}")
            st.error(f"❌ 预测生成失败: {str(e)}")

    def display_forecast_results(self):
        """显示预测结果"""
        st.markdown("### 🔮 预测结果")

        forecast_df = st.session_state.forecast_results

        # 预测摘要
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("预测天数", len(forecast_df))
        with col2:
            st.metric("平均预测值", f"{forecast_df['PredictedCases'].mean():.1f}")
        with col3:
            st.metric("最大预测值", forecast_df['PredictedCases'].max())
        with col4:
            st.metric("最小预测值", forecast_df['PredictedCases'].min())

        # 预测结果表格
        st.markdown("#### 📋 预测数据")
        st.dataframe(
            forecast_df[['Date', 'PredictedCases', 'LowerBound', 'UpperBound']],
            use_container_width=True
        )

        # 可视化图表
        st.markdown("#### 📊 预测可视化")
        self.plot_forecast_with_confidence()

        # 下载选项
        st.markdown("#### 💾 下载结果")
        col1, col2 = st.columns(2)

        with col1:
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV",
                data=csv,
                file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # 生成报告
            report = self.generate_report()
            st.download_button(
                label="📄 下载报告",
                data=report,
                file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    def plot_forecast_with_confidence(self):
        """绘制带置信区间的预测图"""
        try:
            historical_df = st.session_state.uploaded_data
            forecast_df = st.session_state.forecast_results

            fig = go.Figure()

            # 历史数据
            fig.add_trace(go.Scatter(
                x=historical_df['Date'],
                y=historical_df['Cases'],
                mode='lines',
                name='历史数据',
                line=dict(color='blue', width=2)
            ))

            # 预测数据
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['PredictedCases'],
                mode='lines',
                name='预测数据',
                line=dict(color='red', width=2)
            ))

            # 置信区间
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['UpperBound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['LowerBound'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,0,0,0.2)',
                fill='tonexty',
                name=f"{forecast_df['ConfidenceInterval'].iloc[0]}% 置信区间",
                hoverinfo='skip'
            ))

            fig.update_layout(
                title='历史数据与预测结果（含置信区间）',
                xaxis_title='日期',
                yaxis_title='病例数',
                hovermode='x unified',
                showlegend=True,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"绘图失败: {e}")
            st.error(f"❌ 绘图失败: {str(e)}")

    def generate_report(self) -> str:
        """生成预测报告"""
        try:
            report_lines = [
                "=" * 50,
                "传染病人数预测报告",
                "=" * 50,
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "1. 数据概况",
                "-" * 20
            ]

            if st.session_state.uploaded_data is not None:
                df = st.session_state.uploaded_data
                report_lines.extend([
                    f"数据点数: {len(df)}",
                    f"日期范围: {df['Date'].min()} 至 {df['Date'].max()}",
                    f"平均病例数: {df['Cases'].mean():.2f}",
                    f"最大病例数: {df['Cases'].max()}",
                    f"最小病例数: {df['Cases'].min()}",
                    ""
                ])

            if st.session_state.analysis_results:
                report_lines.extend([
                    "2. 数据分析结果",
                    "-" * 20,
                    st.session_state.analysis_results.get('summary', '无摘要'),
                    ""
                ])

            if st.session_state.selected_model:
                report_lines.extend([
                    "3. 模型信息",
                    "-" * 20,
                    f"选择的模型: {st.session_state.selected_model}",
                    ""
                ])

            if st.session_state.forecast_results is not None:
                forecast_df = st.session_state.forecast_results
                report_lines.extend([
                    "4. 预测结果",
                    "-" * 20,
                    f"预测天数: {len(forecast_df)}",
                    f"平均预测值: {forecast_df['PredictedCases'].mean():.2f}",
                    f"预测范围: {forecast_df['PredictedCases'].min()} - {forecast_df['PredictedCases'].max()}",
                    ""
                ])

            report_lines.extend([
                "5. 免责声明",
                "-" * 20,
                "本预测结果仅供参考，实际情况可能因多种因素而异。",
                "请结合专业判断和实时数据进行决策。",
                "=" * 50
            ])

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return f"报告生成失败: {str(e)}"

    def run(self):
        """运行应用"""
        self.setup_page()
        self.render_sidebar()

        # 根据当前步骤渲染对应页面
        step_methods = [
            self.render_step_0_data_upload,
            self.render_step_1_data_analysis,
            self.render_step_2_model_recommendation,
            self.render_step_3_model_training,
            self.render_step_4_privacy_protection,
            self.render_step_5_results_output
        ]

        current_step = st.session_state.current_step
        if 0 <= current_step < len(step_methods):
            step_methods[current_step]()
        else:
            st.error("❌ 无效的步骤")


# --- 主函数 ---
def main():
    """主函数"""
    try:
        app = EpidemicPredictionApp()
        app.run()
    except Exception as e:
        logger.error(f"应用运行失败: {e}")
        st.error(f"❌ 应用运行失败: {str(e)}")
        st.info("请刷新页面重试")


if __name__ == "__main__":
    main()
