import numpy as np
import cv2 
import matplotlib.pyplot as plt 
basketball = cv2.imread('../src/images/basketball_crop.jpg')


# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
ball_mask_opened = cv2.morphologyEx(mask_enhanced,cv2.MORPH_OPEN,kernel)
ball_mask_final=cv2.morphologyEx(ball_mask_opened,cv2.MORPH_CLOSE,kernel)
contours,_=cv2.findContours(ball_mask_final,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
selected_mask=np.zeros_like(ball_mask_final)
final_img=basketball_rgb.copy()
if len(contours) > 0:
    largest_contour = max(contours,key=cv2.contourArea)
    cv2.drawContours(
        selected_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )
    overlay = final_img.copy()
    overlay[selected_mask>0] = (0,255,0)
    final_img = cv2.addWeighted(final_img,0.65,overlay, 0.35,0)

    cv2.drawContours(final_img,[largest_contour],-1,(0,255,0),2)
else:
    print("nono")

plt.figure(figsize=(18,10))
plt.subplot(2,3,1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
plt.subplot(2,3,2), plt.imshow(ball_mask_opened, cmap="gray"), plt.title("Mask after Opening")
plt.subplot(2,3,3), plt.imshow(ball_mask_closed, cmap="gray"), plt.title("Mask after Closing")

plt.subplot(2,3,4), plt.imshow(ball_mask_final, cmap="gray"), plt.title("After opening + closing")
plt.subplot(2,3,5), plt.imshow(selected_mask, cmap="gray"), plt.title("selected Ball mask")
plt.subplot(2,3,6), plt.imshow(final_img, cmap="gray"), plt.title("Basketball contour")


for ax in plt.gcf().axes:
    ax.axis("off")


plt.tight_layout()
plt.show()    
# TODO: Binary mask로 외곽선 검출
# TODO: 최종 농구공 영역 및 외곽선 선택
#readme write  which 


