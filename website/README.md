# 水電工程師個人網站

一個純靜態（HTML / CSS / JavaScript，無相依套件）的個人形象網站，適合水電工程師展示服務項目、施工案例與聯絡方式。

## 檔案結構

| 檔案 | 說明 |
| --- | --- |
| `index.html` | 網站首頁，包含關於我、服務項目、施工案例、服務流程、常見問題與聯絡估價 |
| `styles.css` | 版面與響應式樣式（桌機／手機皆適用） |
| `main.js` | 行動版選單切換、估價表單前端驗證 |

## 本機預覽

```bash
cd website
python -m http.server 8000
```

接著在瀏覽器開啟 <http://localhost:8000>。

## 客製化重點

1. **個人資訊**：`index.html` 中的姓名、證照、電話 `0900-000-000`、Email `service@example.com`、服務區域，請改為真實資料。
2. **服務與案例**：`#services`、`#works` 區塊可依實際承接的工程調整或增減。
3. **報價說明**：`#process` 區塊的出勤費與保固條件請依實際規定修改。
4. **配色**：`styles.css` 最上方的 `:root` 變數可調整主色 `--brand` 與強調色 `--accent`。

## 表單說明

聯絡表單目前僅有前端驗證，送出後不會將資料傳送到任何伺服器。若需實際收件，可串接表單服務（如 Formspree）或自建後端 API，並於 `main.js` 的送出流程中改為呼叫該 API。

## 部署

由於為純靜態網站，可直接部署至 GitHub Pages、Netlify、Cloudflare Pages 等服務，將 `website/` 目錄設為發布根目錄即可。
