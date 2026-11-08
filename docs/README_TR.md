# Türkçe kullanım rehberi

Model, İngilizce rapordan sentetik frontal göğüs radyografisi üretir. Başlangıç
görüntüsü gerekmez. UNet LoRA yalnız görüntü ağındaki uyarlayıcıların eğitilmesi
anlamındadır; rapor her üretimde kullanılır.

Yayın adayı ZIP'ini indirin. IU_Yayin_Dogrulama_Colab.ipynb dosyasını Colab'da
açın; L4, A100 veya H100 GPU seçin. Yüksek RAM gerekmez. Hücreleri sırayla
çalıştırın ve ZIP'i yükleyin. İlk çalıştırmada yaklaşık 4,3 GB taban model
indirilir. Yeniden eğitim yapılmaz.

Defter altı kurgusal örnekle yazılım kontrolü yapar. Bu kontrol klinik başarı
deneyi değildir. Son hücre IU_Release_Dogrulama_Sonucu.zip dosyasını indirir;
hata varsa ayrıntıları da pakete kaydeder.

Kendi raporunuzla aynı ortamda:

    from iu_cxr.pipeline import IUReportGenerator
    generator = IUReportGenerator(str(MODEL_DIR), training_seed=42)
    image, metadata = generator.generate(
        "Frontal chest radiograph. The cardiac silhouette is enlarged.",
        generation_seed=202601
    )
    display(image)

training_seed eğitim tekrarını, generation_seed başlangıç gürültüsünü seçer.
Model Türkçe raporu otomatik çevirmez. Uzun raporların tüm tokenları ayrı CLIP
pencerelerinde işlenir; 16 pencere sınırı aşılırsa açık hata verilir.
Klinik tanı için kullanılmamalı; çıktılar sentetik olarak tanımlanmalıdır.

Bu GitHub deposu kaynak kodunu içerir. Üç IU adaptörü yazarların ayrı
huggingface_model/ klasöründedir; herkese açık model indirme bağlantısı henüz
atanmadı. Colab defteri şu anda yazarların sağladığı tam yayın ZIP'ini gerektirir.
verification/ tamamlanmış CPU ve Colab GPU kontrollerini gösterir. Üç adaptör
başarıyla üretim yaptı. Seed 42 ve uzun kurgusal rapor için özgün kod ile yeni
paketin çıktıları piksel düzeyinde aynıydı. Koşum GPU'yu Tesla T4 olarak bildirdi.
Bu kontrol klinik başarı ölçümü değildir; yeni bir eğitim yapılmadı.
