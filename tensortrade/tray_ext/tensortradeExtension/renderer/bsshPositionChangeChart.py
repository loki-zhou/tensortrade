import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensortrade.env.generic import Renderer
from tensortrade.env.generic import TradingEnv


class PositionChangeChart(Renderer):

    def __init__(self, fig, ax1, ax2):
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

    def render(self, env: TradingEnv):
        self.ax1.clear()
        self.ax2.clear()

        history = pd.DataFrame(env.observer.renderer_history)
        actions = list(history.action)
        price = list(history.price)

        enter_long = {}
        exit_long = {}
        enter_short = {}
        exit_short = {}

        for i in range(len(actions) - 1):
            previous_action = actions[i]
            current_action = actions[i + 1]

            # Actions Interpretation:
                # Enter Long == 0
                # Exit Long == 1
                # Enter Short == 2
                # Exit Short == 3
                # Implement logic for transition from hold to enter long and short positions
            
            # If previous position has changed,
            if current_action != previous_action and current_action != -1:
                
                # Transition from 0 to 1 means that an existing long position is closed
                if current_action == 1 and previous_action == 0:
                    exit_long[i] = price[i]

                # Transition from 2 to 3 means that an existing short position is closed
                if current_action == 3 and previous_action == 2:
                    exit_short[i] == price[i]

                # Transition for 0 to 2 Means that an existing short position is closed and a long position is opened
                if current_action == 0:
                    if previous_action == 2:
                        exit_short[i] = price[i]
                    enter_long[i] = price[i]
                
                # Transition from 0 to 2 means that an existing long position is closed and a short position is opened
                if current_action == 2:
                    if previous_action == 0:
                        exit_long[i] = price[i]
                    enter_short[i] = price[i]

                # Transition from 2 to 1 means that existing short and long positions are closed
                # TODO: Check on the sanity of this = Not Logical Transition
                #if current_action == 1:
                #    if previous_action == 2:
                #        exit_short[i] = price[i]
                #    exit_long[i] = price[i]
        
        enter_long_series = pd.Series(enter_long, dtype = 'object')
        enter_short_series = pd.Series(enter_short, dtype='objecct')  
        exit_long_series = pd.Series(exit_long, dtype='object')
        exit_short_series = pd.Series(exit_short, dtype='object')

        enter_long_series_color = 'g'
        enter_short_series_color = 'b'
        exit_long_series_color = 'r'
        exit_short_series_color = 'y'
        
        self.price_ax.scatter(
            enter_long_series.index,
            enter_long_series.values,
            marker='^',
            color=enter_long_series_color
            )
        '''
        self.price_ax.annotate(
            '{0:.2f}',
            (date, close),
            xytext=(date, close),
            size="large",
            arrowprops=dict(
                arrowstyle='simple',
                facecolor=enter_long_series_color))'''
            
        self.price_ax.scatter(
            exit_long_series.index,
            exit_long_series.values,
            marker='^',
            color=exit_long_series_color
            )
        '''
        self.price_ax.annotate(
            '{0:.2f}',
            (date, close),
            xytext=(date, close),
            size="large",
            arrowprops=dict(
                arrowstyle='simple',
                facecolor=exit_long_series_color))'''

        self.price_ax.scatter(
            enter_short_series.index,
            enter_short_series.values,
            marker='^',
            color=enter_short_series_color
            )
        '''
        self.price_ax.annotate(
            '{0:.2f}',
            (date, close),
            xytext=(date, close),
            size="large",
            arrowprops=dict(
                arrowstyle='simple',
                facecolor=enter_short_series_color))'''

        self.price_ax.scatter(
            exit_short_series.index,
            exit_short_series.values,
            marker='^',
            color=exit_short_series_color
            )
        '''
        self.price_ax.annotate(
            '{0:.2f}',
            (date, close),
            xytext=(date, close),
            size="large",
            arrowprops=dict(
                arrowstyle='simple',
                facecolor=exit_short_series_color))'''

        self.ax1.set_title("Trading Chart")
        self.ax1.plot(np.arange(len(price)), price, label="price", color="orange")

        performance_df = pd.DataFrame().from_dict(env.reward_scheme.net_worth_history)
        performance_df.plot(ax=self.ax2)
        self.ax2.set_title("Net Worth")

    def close(self):
        plt.close()
