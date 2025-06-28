def gradCAM_heatmap(img,model,last_conv_layers_name):
    grad_model=Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output,model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs,prediction=grad_model(img)
        pred_idx=tf.argmax(predition)
        class_channel=prediction[:,pred_idx]

    #to calculate gradients of the target class w.r.t conv output
    grads=tape.gradient(class_channel,conv_outputs)

    # Mean of gradients for each feature map
    pooled_grads=tape.reduce_mean(grads,axis=(0,1,2))

    #weighted sum
    conv_outputs=conv_outputs[0]
    heatmap=conv_outputs @ pooled_grad[:None] #@-
    heatmap=tf.squeeze(heatmap)

    #normalize b/w 0 and 1
    heatmap=tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#SuperImpose Heatmap
def superimpose_heatmap(heapmap,img,alpha=0.4):
    heatmap=cv.resize(heatmap,(img.shape[1],img.shape[0]))

    #convert to RGB
    heatmap=np.unit8(255*heatmap)
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    #superimpose
    superimposed_img=cv2.addWeighted(img,1-alpha,heatmap,alpha,0)
    return superimposed_img
