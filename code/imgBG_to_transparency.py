# 输入图片路径和输出图片路径
input_path = "/_media/logo.png"
output_path = "/_media/logo_transparency.png"

# 打开图片并将背景变为透明
with Image.open(input_path) as im:
    im = im.convert("RGBA")
    data = im.getdata()

    newData = []
    for item in data:
        # 将背景色的像素点变为透明
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    im.putdata(newData)

    # 保存处理后的图片
    im.save(output_path)