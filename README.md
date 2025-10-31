# 完整的動手指南，可在Google Colab GPU上訓練你的神經網路模型

# 介绍

如果你是神經網絡領域的初學者，那麼你可能已經使用過 CPU 訓練模型。好吧，即使你的模型有 100000 個參數也沒關係，訓練模型可能需要幾個小時。但是，如果你的模型有 100 億或 200 億個參數怎麼辦？像 VGG16 這樣常見的 CNN 模型有 1.38 億個參數，因此使用 CPU 訓練這樣的模型將是一個問題，因為它會佔用你許多寶貴的時間。在本文中，我們將討論 GPU 如何為我們解決這個問題，並實際體驗使用 GPU 訓練簡單模型的過程。

# 為什麼 GPU 在某些任務中比 CPU 更優？

而不是我介紹它的好處，相信我這個影片會給你一個更清楚的概念。

來源：[https://www.youtube.com/watch?v=ZrJeYFxpUyQ&t=7s](https://www.youtube.com/watch?v=ZrJeYFxpUyQ&t=7s)

現在你可能已經有一些想法了吧？是的，GPU 這種大規模並行計算能力極大地幫助我們提升複雜神經網絡模型的效能，並減少訓練時間。GPU 包含大量內建的較小核心，有助於完成這些任務。

在神經網絡中，最基本的運算是矩陣乘法，而 GPU 對這個任務非常擅長，它就像專門研究矩陣乘法的專業數學家一樣處理這些計算。GPU 相較於 CPU 的其他一些優勢包括：

* 它具有更大的記憶體頻寬。
* 較小的 L1 和 L2 快取有助於更快速地存取快取記憶體。
* 為了高效利用 GPU 的多核心，我們使用了 CUDA 編程模型。在 PyTorch 中，執行 CUDA 操作要容易得多。

但請切記，GPU 並不會在所有用途上取代 CPU，因為在主程式仍在運行時，GPU 僅作為 CPU 的額外協助者，透過幫助執行給定應用程式的並行重複計算來貢獻效能。

GPU 相較於 CPU 更適合的一些應用還有：

* **影片渲染**——由於高計算能力和記憶體頻寬，它可以有效地渲染影片。
* **加密貨幣挖礦**——最初使用 CPU 進行加密貨幣挖礦，但由於功耗高且處理能力有限，效果不理想。目前已有專用的 GPU 可用於挖礦，例如 Nvidia GeForce RTX 2080 Ti。

與其只停留在理論，不如讓我們透過在 Google Colab notebook 上使用 GPU 訓練模型來實際操作看看。

# 在 Google Colab 中在 GPU 上訓練神經網絡模型
使用 Google Colab 環境，我們可以免費使用 **NVIDIA Tesla K80** GPU。但請記住，你只能連續使用它 12 小時，之後可能無法在特定時間內再次使用，除非你購買 Colab Pro。

我們將使用 **MNIST 手寫數字分類資料集** 作為範例。我們的任務是訓練一個模型，使其能夠將給定的手寫數字圖像正確分類到對應的標籤。因此，在 GPU 上訓練模型時，你需要注意的主要步驟包括：

* 設定運行時類型。
* 定義一個可以在 GPU 和 CPU 之間切換的函數。
* 將資料集和模型加載到 GPU 中。

# 步驟 1：設定 Google Colab 筆記本
在建立新筆記本後，第一步是將 **運行時類型** 設定為 GPU。
<img width="769" height="659" alt="螢幕擷取畫面 2025-11-01 015349" src="https://github.com/user-attachments/assets/cf1fce9d-031d-4c21-aa41-5c33129f8267" />
<img width="561" height="521" alt="螢幕擷取畫面 2025-11-01 015421" src="https://github.com/user-attachments/assets/91077c54-5e39-4e3e-a7ee-c6e37270977e" />

# 步驟 2：載入必要的程式庫
<img width="500" height="329" alt="螢幕擷取畫面 2025-11-01 015555" src="https://github.com/user-attachments/assets/ef6ff2c4-af75-429a-a7b0-78f633615db4" />

# 步驟 3：建立訓練與驗證資料集

<img width="606" height="290" alt="image" src="https://github.com/user-attachments/assets/e65e6a1b-92d0-452f-af70-f41401a572b3" />

# 步驟 4：批次載入訓練與驗證資料集

<img width="781" height="91" alt="image" src="https://github.com/user-attachments/assets/347777ba-593f-4d26-a62a-cfd8a15db78f" />
當你想將 CPU 上載入的資料集推送到 GPU 時，設定 pin_memory = True 會使兩者之間的資料傳輸速度更快。

# 步驟 5：建立 MnistModel 類別

<img width="950" height="732" alt="image" src="https://github.com/user-attachments/assets/2af75220-a878-4edd-b830-563a0c45870e" />

使用該類別，我們可以建立需要訓練的模型。在將模型和資料加載到 GPU 之前，我們先檢查 GPU 是否可用：
<img width="288" height="82" alt="image" src="https://github.com/user-attachments/assets/faeb3371-161c-4e3a-94a8-7e93f8d8ff38" />

如果一切順利，你可能會得到輸出為 **True**。但由於你沒有使用 Colab Pro，如果你連續使用一段時間，有時 GPU 可能無法使用。
因此，在下一步中，我們將建立一個可以在 GPU 和 CPU 之間切換的函數，讓程式在 GPU 不可用時自動切換到 CPU。

# 步驟 6：建立一個輔助函數，用於在 CPU 和 GPU 之間切換

<img width="503" height="184" alt="image" src="https://github.com/user-attachments/assets/f40b50b6-5c7d-4aa1-963f-5f422d31adf6" />
現在即使 GPU 不可用也沒問題，因為系統會自動切換到 CPU 進行訓練，但訓練所需的時間會較長。

# 步驟 7：定義將資料或模型移動到 GPU 的函數

<img width="553" height="127" alt="image" src="https://github.com/user-attachments/assets/71057bb7-0374-410b-b2fa-4e4b396de24e" />

# 步驟 8：建立 DeviceDataLoader 類別

<img width="499" height="210" alt="image" src="https://github.com/user-attachments/assets/c259f127-6a6e-46bd-8f20-3b3abd63ea28" />

使用 DeviceDataLoader 類別，我們可以建立幫助將 train_loader 和 val_loader（在步驟 4 中定義）中的資料移動到 GPU 的物件。

#步驟 9：定義用於訓練與驗證模型的函數

<img width="645" height="371" alt="image" src="https://github.com/user-attachments/assets/4d002ae2-a55d-4eaa-a8ca-b7b53e82cf1c" />

# 步驟 10：建立 MNISTModel 的實例，並將其移動到 GPU

<img width="649" height="173" alt="image" src="https://github.com/user-attachments/assets/f396faa0-a3e4-4ec9-a0c0-22d9e9071e62" />

請注意，在建立模型後，你必須將模型移動到 GPU，否則我們已經移動到 GPU 的資料將無法與仍在 CPU 的模型配合使用。

# 步驟 11：訓練與驗證模型

<img width="789" height="425" alt="image" src="https://github.com/user-attachments/assets/4893572c-60f8-4eac-861b-c023e2bc33c1" />

**中間遇到accuracy沒有定義的問題，所以重新定義一個accuracy**
<img width="657" height="86" alt="image" src="https://github.com/user-attachments/assets/55134ad1-cf0b-4f85-85dc-9465e0cc5b39" />


# 結論
現在我希望你可能對於使用 GPU 訓練模型有了更好的理解，以及在 GPU 訓練階段需要記住的三個重要步驟。

為了讓你了解更多，最近由萊斯大學的一組電腦科學家創建了一種稱為 SLIDE（亞線性深度學習引擎）的新演算法。該演算法背後的主要思想是減少在反向傳播中進行的無用計算。這種高效的演算法僅使用 CPU 來訓練深度學習模型，而不依賴於硬體加速器。
