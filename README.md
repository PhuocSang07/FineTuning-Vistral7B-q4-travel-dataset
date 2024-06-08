# FineTuning-Chatbot-Vistral7B-With-Travel-Dataset

## General

Fine tuning model Viet-Mistral 7B on Travel Vietnamese QA dataset.

  

## About dataset

- The dataset was created by Gemini-1.5 with topics about Vietnam tourism such as famous places, costs travel, local food, culture and festivals, etc. . .

- Includes 4500 rows with 2 columns: *ques*, *ans*.

| ques                                                      | ans                                                                                                                  |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| "Địa điểm du lịch nổi tiếng ở Đà Nẵng là gì?"             | "Đà Nẵng nổi tiếng với Bà Nà Hills, Sơn Trà Peninsula, và Cầu Rồng."                                                 |
| "Có nên thử các món ăn đặc trưng của Đà Nẵng không?"      | "Có, bạn nên thử một số món như bánh xèo, mì Quảng, và bún chả cá Đà Nẵng.",                                         |
| "Những hoạt động nào nên tham gia ở Đà Nẵng?"             | "Bạn có thể tham gia các hoạt động như leo núi, thăm các công viên giải trí, và tắm biển ở Đà Nẵng."                 |
| "Thời tiết thích hợp nhất để du lịch Đà Nẵng là khi nào?" | "Thời tiết tốt nhất là từ tháng 2 đến tháng 8, khi không có mưa và nhiệt độ không quá cao."                          |
| "Phương tiện di chuyển phổ biến ở Đà Nẵng là gì?"         | "Phương tiện di chuyển phổ biến ở Đà Nẵng bao gồm taxi, xe máy và xe đạp đều có thể thuê."                           |
| "Những địa điểm tham quan nổi tiếng ở Quảng Nam là gì?"   | "Quảng Nam có các địa điểm như Hội An Ancient Town, My Son Sanctuary và Bảo tàng lịch sử Quảng Nam."                 |
| "Có những món ăn đặc trưng nào ở Quảng Nam?"              | "Một số món ăn đặc trưng của Quảng Nam là cơm gà Hội An, bánh mì Phượng và bánh đập."                                |
| "Những hoạt động nào nên thử khi du lịch Quảng Nam?"      | "Bạn có thể tham gia các hoạt động như tham quan phố cổ Hội An, trải nghiệm làm nông dân và thăm làng gốm Thanh Hà." |

## Model

Vistral (Viet Mistral) is a model developed from model Mistral for Vietnamese QA problems.
*Model is loaded in 4bit format.*

```
bnb_config = BitsAndBytesConfig(
	load_in_4bit= True,
	bnb_4bit_quant_type= "nf4",
	bnb_4bit_compute_dtype= torch.bfloat16,
	bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
	"Viet-Mistral/Vistral-7B-Chat",
	load_in_4bit=True,
	quantization_config=bnb_config,
	torch_dtype=torch.bfloat16,
	device_map="auto",
	trust_remote_code=True,
)
```

Load model
```
# Load base model 4bit
# ........
model.config.use_cache = False #silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.bos_token, tokenizer.eos_token
```

## Run history

| train/epoch                    | ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███ |
| ------------------------------ | ---------------------------------------- |
| train/global_step              | ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███ |
| train/learning_rate            | ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ |
| train/loss                     | █▅▅▄▄▄▅▃▄▄▃▃▃▁▂▂▂▂▂▂▁▂▂▂▂▂▂▁▂▁▁▁▁▁▁▂▁▁▂▁ |
| train/total_flos               | ▁                                        |
| train/train_loss               | ▁                                        |
| train/train_runtime            | ▁                                        |
| train/train_samples_per_second | ▁                                        |
| train/train_steps_per_second   | ▁                                        |

## Run summary
| train/epoch                    | 3.0                |
| ------------------------------ | ------------------ |
| train/global_step              | 189                |
| train/learning_rate            | 0.0002             |
| train/loss                     | 0.5583             |
| train/total_flos               | 7456187001864192.0 |
| train/train_loss               | 0.83275            |
| train/train_runtime            | 2447.9379          |
| train/train_samples_per_second | 1.226              |
| train/train_steps_per_second   | 0.077              |