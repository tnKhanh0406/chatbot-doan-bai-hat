import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random

# Hàm tự động convert entity
def convert_data(train_data):
    new_data = []
    for text, annot in train_data:
        entities = []
        for entity_text, label in annot["entities"]:
            start = text.lower().find(entity_text.lower())
            if start == -1:
                print(f"⚠️ Warning: '{entity_text}' not found in '{text}'. Skipping entity.")
                continue
            end = start + len(entity_text)
            entities.append((start, end, label))
        if entities:
            new_data.append((text, {"entities": entities, "cats": annot["cats"]}))
        else:
            print(f"⚠️ Warning: No valid entities for '{text}'. Skipping sample.")
    return new_data

# Dữ liệu mẫu
SAMPLE_DATA = [
    ("Tôi muốn biết bài hát có câu: em vẫn muốn yêu anh thêm lần nữa", {
        "entities": [("em vẫn muốn yêu anh thêm lần nữa", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Anh vẫn đặt trái tim lên bàn Nếu em không màng", {
        "entities": [("Anh vẫn đặt trái tim lên bàn Nếu em không màng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Tôi muốn biết bài hát có câu: chúng ta không thuộc về nhau của ca sĩ Sơn Tùng MTP", {
        "entities": [("chúng ta không thuộc về nhau", "LYRICS"), ("Sơn Tùng MTP", "SINGER")], "cats": {"find_song": 1.0}
    }),
    ("Bài hát nào có câu: Bọn mình đi đâu đi anh ơi đi đu đưa đi Lúc đi hết mình lúc về hết buồn", {
        "entities": [("Bọn mình đi đâu đi anh ơi đi đu đưa đi Lúc đi hết mình lúc về hết buồn", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hà Anh Tuấn hát câu: Từng cơn mưa hắt hiu bên ngoài sông thưa Lắm khi mưa làm hồn ta nhớ mãi ngày qua", {
        "entities": [("Hà Anh Tuấn", "SINGER"), ("Từng cơn mưa hắt hiu bên ngoài sông thưa Lắm khi mưa làm hồn ta nhớ mãi ngày qua", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: Những cánh hoa phai tàn thật nhanh, em có baу xa, em có đi xa mãi", {
        "entities": [("Ɲhững cánh hoa phai tàn thật nhanh, em có baу xa, em có đi xa mãi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Mỹ Tâm có bài nào với lời: đừng hỏi em vì sao?", {
        "entities": [("Mỹ Tâm", "SINGER"), ("đừng hỏi em vì sao", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: Bờ vai ai mang vác tấm hàng Mồ hôi tuôn rơi những tháng năm hao gầy", {
        "entities": [("Bờ vai ai mang vác tấm hàng Mồ hôi tuôn rơi những tháng năm hao gầy", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Sơn Tùng MTP có bài nào với lời: Hãy trao cho anh thứ anh đang mong chờ", {
        "entities": [("Sơn Tùng MTP", "SINGER"), ("Hãy trao cho anh thứ anh đang mong chờ", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: đừng như thói quen là của ai?", {
        "entities": [("đừng như thói quen", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Đen Vâu hát câu: Mấy đứa trẻ đi lên trường, đội trên đầu là đóa mây trắng Chân đạp lên mặt trời, môi thì cười và má hây nắng", {
        "entities": [("Đen Vâu", "SINGER"), ("Mấy đứa trẻ đi lên trường, đội trên đầu là đóa mây trắng Chân đạp lên mặt trời, môi thì cười và má hây nắng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài nào có câu: Bố em hút rất nhiều thuốc Mẹ em khóc mắt lệ nhoà", {
        "entities": [("Bố em hút rất nhiều thuốc Mẹ em khóc mắt lệ nhoà", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hồ Ngọc Hà có bài nào với lời: đừng để em xa anh?", {
        "entities": [("Hồ Ngọc Hà", "SINGER"), ("đừng để em xa anh", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài hát nào có câu: mơ một giấc mơ thật dài?", {
        "entities": [("mơ một giấc mơ thật dài", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Sơn Tùng hát câu: chạy ngay đi trước khi mọi điều dần tồi tệ hơn", {
        "entities": [("Sơn Tùng", "SINGER"), ("chạy ngay đi trước khi mọi điều dần tồi tệ hơn", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: em là ai giữa cơn đau này là của ai?", {
        "entities": [("em là ai giữa cơn đau này", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Mỹ Tâm có bài nào với lời: hãy nói anh nghe một lời?", {
        "entities": [("Mỹ Tâm", "SINGER"), ("hãy nói anh nghe một lời", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: anh làm gì sai để em ra đi?", {
        "entities": [("anh làm gì sai để em ra đi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bích Phương hát câu: đi đu đưa đi là bài nào?", {
        "entities": [("Bích Phương", "SINGER"), ("đi đu đưa đi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("chạy ngay đi trước khi mọi điều dần tồi tệ hơn", {
        "entities": [("chạy ngay đi trước khi mọi điều dần tồi tệ hơn", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("tôi muốn biết bài hát: Người ơi hãy tin anh này và em đến đây đi Ngàn hoa lá đang thay màu này em thích hoa gì của ca sĩ DatKaa", {
        "entities": [("Người ơi hãy tin anh này và em đến đây đi Ngàn hoa lá đang thay màu này em thích hoa gì", "LYRICS"), ("DatKaa", "SINGER")], "cats": {"find_song": 1.0}
    })
]

SAMPLE_DATA += [
    ("Bài nào có câu: Em bỏ quên mình giữa trời đầy gió", {
        "entities": [("Em bỏ quên mình giữa trời đầy gió", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Sơn Tùng MTP hát câu: Muộn rồi mà sao còn", {
        "entities": [("Sơn Tùng MTP", "SINGER"), ("Muộn rồi mà sao còn", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát bài có câu: Cơn mưa ngang qua mang em đi xa", {
        "entities": [("Cơn mưa ngang qua mang em đi xa", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Erik có bài nào với lời: Sau tất cả mình lại trở về với nhau", {
        "entities": [("Erik", "SINGER"), ("Sau tất cả mình lại trở về với nhau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài hát nào có câu: Tháng tư là lời nói dối của em", {
        "entities": [("Tháng tư là lời nói dối của em", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: Gặp nhưng không ở lại là của ai?", {
        "entities": [("Gặp nhưng không ở lại", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: Nơi tình yêu bắt đầu?", {
        "entities": [("Nơi tình yêu bắt đầu", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bích Phương có bài nào với lời: Bao giờ lấy chồng?", {
        "entities": [("Bích Phương", "SINGER"), ("Bao giờ lấy chồng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài nào có câu: Bức tranh từ nước mắt", {
        "entities": [("Bức tranh từ nước mắt", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Đen Vâu hát câu: Mang tiền về cho mẹ", {
        "entities": [("Đen Vâu", "SINGER"), ("Mang tiền về cho mẹ", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài hát có câu: Ngày mai người ta lấy chồng rồi", {
        "entities": [("Ngày mai người ta lấy chồng rồi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: Giữ em đi là của ca sĩ nào?", {
        "entities": [("Giữ em đi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: Vì em là của anh", {
        "entities": [("Vì em là của anh", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Noo Phước Thịnh có bài nào với lời: Cause you're my baby", {
        "entities": [("Noo Phước Thịnh", "SINGER"), ("Cause you're my baby", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài nào có câu: Trót yêu một người", {
        "entities": [("Trót yêu một người", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: Em gái mưa của Hương Tràm", {
        "entities": [("Em gái mưa", "LYRICS"), ("Hương Tràm", "SINGER")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: Bước qua nhau nhưng sao lòng đau?", {
        "entities": [("Bước qua nhau nhưng sao lòng đau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hà Anh Tuấn có bài nào với lời: Tháng mấy em nhớ anh?", {
        "entities": [("Hà Anh Tuấn", "SINGER"), ("Tháng mấy em nhớ anh", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài hát nào có câu: Anh cứ đi đi và đừng quay về", {
        "entities": [("Anh cứ đi đi và đừng quay về", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: Anh ơi ở lại là của ai?", {
        "entities": [("Anh ơi ở lại", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Mỹ Tâm hát câu: Đâu chỉ riêng em", {
        "entities": [("Mỹ Tâm", "SINGER"), ("Đâu chỉ riêng em", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: Ta còn yêu nhau chăng?", {
        "entities": [("Ta còn yêu nhau chăng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hồ Ngọc Hà có bài nào với lời: Cả một trời thương nhớ", {
        "entities": [("Hồ Ngọc Hà", "SINGER"), ("Cả một trời thương nhớ", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài nào có câu: Gửi anh xa nhớ", {
        "entities": [("Gửi anh xa nhớ", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Sơn Tùng hát câu: Chúng ta của hiện tại", {
        "entities": [("Sơn Tùng", "SINGER"), ("Chúng ta của hiện tại", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lời bài hát: Một bước yêu vạn dặm đau là của ai?", {
        "entities": [("Một bước yêu vạn dặm đau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ai hát câu: Em không sai, chúng ta sai", {
        "entities": [("Em không sai, chúng ta sai", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Noo Phước Thịnh có bài nào với lời: Gạt đi nước mắt", {
        "entities": [("Noo Phước Thịnh", "SINGER"), ("Gạt đi nước mắt", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bài nào có câu: Hơn cả yêu", {
        "entities": [("Hơn cả yêu", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Erik hát câu: Chạm đáy nỗi đau", {
        "entities": [("Erik", "SINGER"), ("Chạm đáy nỗi đau", "LYRICS")], "cats": {"find_song": 1.0}
    })
]

SAMPLE_DATA += [
    ("Anh ơi anh à, đừng buông tay em ra", {
        "entities": [("Anh ơi anh à, đừng buông tay em ra", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ngày chưa giông bão, ta vui biết bao nhiêu", {
        "entities": [("Ngày chưa giông bão, ta vui biết bao nhiêu", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Có chàng trai viết lên cây lời yêu thương cô gái ấy", {
        "entities": [("Có chàng trai viết lên cây lời yêu thương cô gái ấy", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Em là ai từ đâu bước đến nơi đây dịu dàng chân phương", {
        "entities": [("Em là ai từ đâu bước đến nơi đây dịu dàng chân phương", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Người từng nói sẽ mãi bên tôi giờ đâu rồi", {
        "entities": [("Người từng nói sẽ mãi bên tôi giờ đâu rồi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Mình chia tay đi cho nhẹ lòng nhau", {
        "entities": [("Mình chia tay đi cho nhẹ lòng nhau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bên em là biển rộng, bên em là đại dương", {
        "entities": [("Bên em là biển rộng, bên em là đại dương", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Từng yêu nhau thế thôi, thế thôi là hết", {
        "entities": [("Từng yêu nhau thế thôi, thế thôi là hết", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bông hoa đẹp nhất là bông hoa gần bên em", {
        "entities": [("Bông hoa đẹp nhất là bông hoa gần bên em", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Người lạ ơi, cho tôi mượn bờ vai", {
        "entities": [("Người lạ ơi, cho tôi mượn bờ vai", "LYRICS")], "cats": {"find_song": 1.0}
    })
]
SAMPLE_DATA += [
    ("Sơn Tùng MTP hát câu: Chúng ta của hiện tại chẳng thể quay lại", {
        "entities": [("Sơn Tùng MTP", "SINGER"), ("Chúng ta của hiện tại chẳng thể quay lại", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Mỹ Tâm có bài nào với lời: Đừng hỏi em vì sao giấc mơ tàn", {
        "entities": [("Mỹ Tâm", "SINGER"), ("Đừng hỏi em vì sao giấc mơ tàn", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Đen Vâu hát câu: Mang tiền về cho mẹ đừng mang ưu phiền về cho mẹ", {
        "entities": [("Đen Vâu", "SINGER"), ("Mang tiền về cho mẹ đừng mang ưu phiền về cho mẹ", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hà Anh Tuấn hát câu: Tháng tư là lời nói dối của em", {
        "entities": [("Hà Anh Tuấn", "SINGER"), ("Tháng tư là lời nói dối của em", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("ERIK có bài nào với lời: Sau tất cả mình lại trở về với nhau", {
        "entities": [("ERIK", "SINGER"), ("Sau tất cả mình lại trở về với nhau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Noo Phước Thịnh hát câu: Gạt đi nước mắt để quên một người", {
        "entities": [("Noo Phước Thịnh", "SINGER"), ("Gạt đi nước mắt để quên một người", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bích Phương có bài nào với lời: Em chào anh, anh đi đâu đấy?", {
        "entities": [("Bích Phương", "SINGER"), ("Em chào anh, anh đi đâu đấy?", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Trúc Nhân hát câu: Sáng mắt chưa, sáng mắt chưa", {
        "entities": [("Trúc Nhân", "SINGER"), ("Sáng mắt chưa, sáng mắt chưa", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("MIN có bài nào với lời: Có ai thương em như anh đã từng", {
        "entities": [("MIN", "SINGER"), ("Có ai thương em như anh đã từng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("JustaTee hát câu: Thằng điên khi yêu em", {
        "entities": [("JustaTee", "SINGER"), ("Thằng điên khi yêu em", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Soobin Hoàng Sơn có bài nào với lời: Phía sau một cô gái là khoảng trời cô đơn", {
        "entities": [("Soobin Hoàng Sơn", "SINGER"), ("Phía sau một cô gái là khoảng trời cô đơn", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hương Tràm hát câu: Em gái mưa đứng dưới mưa", {
        "entities": [("Hương Tràm", "SINGER"), ("Em gái mưa đứng dưới mưa", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Mr Siro có bài nào với lời: Lặng thầm yêu ai đó", {
        "entities": [("Mr Siro", "SINGER"), ("Lặng thầm yêu ai đó", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hoàng Dũng hát câu: Nàng thơ của anh là ai?", {
        "entities": [("Hoàng Dũng", "SINGER"), ("Nàng thơ của anh là ai?", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Chi Pu có bài nào với lời: Anh ơi ở lại, đừng về", {
        "entities": [("Chi Pu", "SINGER"), ("Anh ơi ở lại, đừng về", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("AMEE hát câu: Anh nhà ở đâu thế?", {
        "entities": [("AMEE", "SINGER"), ("Anh nhà ở đâu thế?", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Đức Phúc có bài nào với lời: Hơn cả yêu là em", {
        "entities": [("Đức Phúc", "SINGER"), ("Hơn cả yêu là em", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Ngọt hát câu: Em dạo này ổn không?", {
        "entities": [("Ngọt", "SINGER"), ("Em dạo này ổn không?", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Tóc Tiên có bài nào với lời: Vũ điệu cồng chiêng vang lên", {
        "entities": [("Tóc Tiên", "SINGER"), ("Vũ điệu cồng chiêng vang lên", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Tiên Tiên hát câu: Vì tôi còn sống", {
        "entities": [("Tiên Tiên", "SINGER"), ("Vì tôi còn sống", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Trịnh Thăng Bình có bài nào với lời: Người ấy từng là tất cả", {
        "entities": [("Trịnh Thăng Bình", "SINGER"), ("Người ấy từng là tất cả", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hòa Minzy hát câu: Rời bỏ những tháng năm bên nhau", {
        "entities": [("Hòa Minzy", "SINGER"), ("Rời bỏ những tháng năm bên nhau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bảo Anh có bài nào với lời: Sống xa anh chẳng dễ dàng", {
        "entities": [("Bảo Anh", "SINGER"), ("Sống xa anh chẳng dễ dàng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Vũ Cát Tường hát câu: Yêu xa là thế", {
        "entities": [("Vũ Cát Tường", "SINGER"), ("Yêu xa là thế", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Bùi Anh Tuấn có bài nào với lời: Nơi tình yêu bắt đầu", {
        "entities": [("Bùi Anh Tuấn", "SINGER"), ("Nơi tình yêu bắt đầu", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("OnlyC hát câu: Yêu là tha thu", {
        "entities": [("OnlyC", "SINGER"), ("Yêu là tha thu", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Karik có bài nào với lời: Người lạ ơi, cho tôi mượn bờ vai", {
        "entities": [("Karik", "SINGER"), ("Người lạ ơi, cho tôi mượn bờ vai", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Quang Hùng MasterD hát câu: Dễ đến dễ đi", {
        "entities": [("Quang Hùng MasterD", "SINGER"), ("Dễ đến dễ đi", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lou Hoàng có bài nào với lời: Mình là gì của nhau", {
        "entities": [("Lou Hoàng", "SINGER"), ("Mình là gì của nhau", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hoaprox hát câu: Ngẫu hứng", {
        "entities": [("Hoaprox", "SINGER"), ("Ngẫu hứng", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Kay Trần có bài nào với lời: Ý em sao", {
        "entities": [("Kay Trần", "SINGER"), ("Ý em sao", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Phan Mạnh Quỳnh hát câu: Vợ người ta", {
        "entities": [("Phan Mạnh Quỳnh", "SINGER"), ("Vợ người ta", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Trọng Hiếu có bài nào với lời: Anh đổ em chưa?", {
        "entities": [("Trọng Hiếu", "SINGER"), ("Anh đổ em chưa?", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Suni Hạ Linh hát câu: Em đã biết", {
        "entities": [("Suni Hạ Linh", "SINGER"), ("Em đã biết", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Lynk Lee có bài nào với lời: Tạm biệt nhé", {
        "entities": [("Lynk Lee", "SINGER"), ("Tạm biệt nhé", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Nguyễn Trần Trung Quân hát câu: Màu nước mắt", {
        "entities": [("Nguyễn Trần Trung Quân", "SINGER"), ("Màu nước mắt", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hương Giang có bài nào với lời: Anh đang ở đâu đấy anh", {
        "entities": [("Hương Giang", "SINGER"), ("Anh đang ở đâu đấy anh", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Hiền Hồ hát câu: Có như không có", {
        "entities": [("Hiền Hồ", "SINGER"), ("Có như không có", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Nếu như một câu nói có thể khiến anh vui Nói thêm một câu nữa có khi khiến anh buồn Nếu em làm như thế trông em có hâm không?", {
        "entities": [("Nếu như một câu nói có thể khiến anh vui Nói thêm một câu nữa có khi khiến anh buồn Nếu em làm như thế trông em có hâm không?", "LYRICS")], "cats": {"find_song": 1.0}
    }),
    ("Một bậc quân vương mang trong con tim hình hài đất nước Ngỡ như dân an ta sẽ chẳng bao giờ buồn", {
        "entities": [("Một bậc quân vương mang trong con tim hình hài đất nước Ngỡ như dân an ta sẽ chẳng bao giờ buồn", "LYRICS")], "cats": {"find_song": 1.0}
    })
]

# Convert dữ liệu
TRAIN_DATA = convert_data(SAMPLE_DATA)

# Load model blank tiếng Việt
nlp = spacy.blank("vi")

# Thêm pipeline NER
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")
ner.add_label("SINGER")
ner.add_label("LYRICS")

# Thêm pipeline textcat_multilabel
if "textcat_multilabel" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat_multilabel")
else:
    textcat = nlp.get_pipe("textcat_multilabel")
textcat.add_label("find_song")

# Bắt đầu train
optimizer = nlp.begin_training()
n_iter = 100

for itn in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.5))
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        nlp.update(examples, drop=0.4, losses=losses)
    print(f"Vòng lặp {itn}, Losses: {losses}")

# Lưu mô hình
nlp.to_disk("model_intent_ner")
print("✅ Mô hình đã lưu vào 'model_intent_ner'")

# Kiểm tra mô hình
test_texts = [
    "Tôi muốn biết bài hát có câu: em vẫn muốn yêu anh thêm lần nữa",
    "chạy ngay đi trước khi mọi điều dần tồi tệ hơn",
    "Tôi muốn biết bài hát có câu: chúng ta chẳng còn gì để nói của ca sĩ Sơn Tùng",
    "Lời bài hát: Những cánh hoa phai tàn thật nhanh, em có baу xa, em có đi xa mãi"
]
print("\nKết quả kiểm tra:")
for test_text in test_texts:
    doc = nlp(test_text)
    print(f"\nCâu: {test_text}")
    print(f"Ý định: {doc.cats}")
    for ent in doc.ents:
        print(f"{ent.text} -> {ent.label_}")