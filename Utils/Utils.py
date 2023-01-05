import plotly.graph_objects as go

def animate_plot(hisotry, Plot_mode='DICE', title=None, y_range_loss_range=2, delay=50, Darmode = False, SHOW_button = False):
    model = hisotry
    model.rename(columns={model.columns[0]: 'epoch'}, inplace=True)
    if max(model['epoch']) < len(model['epoch']):
        model['epoch'] = model['epoch'].apply(lambda x: x+1)

    if Plot_mode == "DICE":
        trace1_col = 'dice_coef'
        trace2_col = 'val_dice_coef'
        y_range_max_range = 1
        #model['accuracy'] = model['accuracy'].apply(lambda x: round(x, 4))
        #model['val_accuracy'] = model['val_accuracy'].apply(lambda x: round(x, 4))
        if title == None:
            title = "Model Accuracy history"
    elif Plot_mode == "LOSS":
        trace1_col = 'loss'
        trace2_col = 'val_loss'
        y_range_max_range = y_range_loss_range
        #model['loss'] = model['loss'].apply(lambda x: round(x, 4))
        #model['val_loss'] = model['val_loss'].apply(lambda x: round(x, 4))
        if title == None:
            title = "Model Loss history"

    trace1 = go.Scatter(x=hisotry[hisotry.columns[0]],
                        y=hisotry[trace1_col], name=trace1_col)
    trace2 = go.Scatter(x=hisotry[hisotry.columns[0]],
                        y=hisotry[trace2_col], name=trace2_col)
    if SHOW_button:
        fig = go.Figure(
            data=[trace1, trace2],
            #data = [go.scatter(x=model['epoch'], y=[model['loss'],model['val_loss']], name='val_loss')],
            layout=go.Layout(
                # template='plotly_dark',
                xaxis=dict(range=[1, hisotry[hisotry.columns[0]]+1], autorange=False),
                yaxis=dict(range=[0, y_range_max_range], autorange=False),
                # * buttons
                showlegend=True,
                hovermode='x unified',
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": delay, "redraw": False},
                                            "fromcurrent": True,

                                            "transition": {"duration": 50, "easing": "quadratic-in-out"}
                                            }]),

                            dict(label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}])
                            ])]),
            # * frames
            frames=[dict(data=[dict(type='scatter', x=hisotry['epoch'][:k+1], y=hisotry[trace1_col][:k+1]),
                            dict(type='scatter', x=hisotry['epoch'][:k+1], y=hisotry[trace2_col][:k+1])], traces=[0, 1],
                        )for k in range(2, len(hisotry['epoch'])+1)],)
        
    else:
        fig = go.Figure(
            data=[trace1, trace2],
            #data = [go.scatter(x=model['epoch'], y=[model['loss'],model['val_loss']], name='val_loss')],
            layout=go.Layout(
                # template='plotly_dark',
                xaxis=dict(range=[1, hisotry[hisotry.columns[0]]], autorange=False),
                yaxis=dict(range=[0, y_range_max_range], autorange=False),
                # * buttons
                showlegend=True,
                hovermode='x unified',
            ))
        
    # fig.update_layout(xaxis=dict(rangeselector = dict(font = dict( color = "black"))))
    # update button color

    # fig.layout.update(template='plotly_dark')
    fig.update_layout(
        title=title,
        xaxis_title="Epochs",
        yaxis_title="Loss",
        title_x=0.5)
    fig.update_layout(title = title)
    
    if Darmode:
        fig.update_layout(template='plotly_dark')
    fig.show()
