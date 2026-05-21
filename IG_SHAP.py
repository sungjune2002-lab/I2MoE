import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 설명 가능성 AI(XAI) 라이브러리 임포트
from captum.attr import IntegratedGradients
import shap

# ==========================================
# 1. 테스트용 임시 멀티모달 모델 정의 (Flex-MoE 구조 모사)
# ==========================================
class PrototypeMultiModalModel(nn.Module):
    def __init__(self, clinical_dim=10, image_dim=8, num_classes=3):
        super(PrototypeMultiModalModel, self).__init__()
        # 임상 데이터 인코더
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )
        # 이미지 데이터 인코더 (더미 특징 입력단)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 32),
            nn.ReLU()
        )
        # 결합 및 분류기
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, clinical_data, image_data):
        # 두 모달리티를 각각 인코딩 후 결합(Concat)
        c_feat = self.clinical_encoder(clinical_data)
        i_feat = self.image_encoder(image_data)
        x = torch.cat((c_feat, i_feat), dim=1)
        logits = self.classifier(x)
        return logits


# ==========================================
# 2. 결과 시각화 및 파일 저장 헬퍼 함수
# ==========================================
def plot_importance(importance_array, feature_names, title, full_save_path):
    mean_importance = np.mean(np.abs(importance_array), axis=0)
    indices = np.argsort(mean_importance)

    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.barh(range(len(indices)), mean_importance[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Mean Absolute Attribution')
    plt.tight_layout()
    
    # 전달받은 전체 절대경로로 차트 이미지 저장
    plt.savefig(full_save_path)
    plt.close()
    print(f"시각화 차트 저장 완료: {full_save_path}")


# ==========================================
# 3. 메인 실행 루프 (5단계 시뮬레이션)
# ==========================================
def main():
    print("[5단계 프로토타입] 외부 파일 배제 모드로 실험을 시작합니다.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 규격 설정 (임상 변수 10개, 이미지 특징 8개로 단순화)
    clinical_dim = 10
    image_dim = 8
    num_classes = 3  # 0: 정상, 1: 경도인지장애, 2: 치매
    num_samples = 5  # 테스트해 볼 임시 환자 수

    # 1) 가짜 데이터(Dummy Data) 즉석 생성
    print("1. 실제 ADNI 파일을 대체할 가짜(Dummy) 데이터 생성 중...")
    dummy_clinical = torch.randn(num_samples, clinical_dim).to(device)
    dummy_image = torch.randn(num_samples, image_dim).to(device)
    dummy_labels = torch.randint(0, num_classes, (num_samples,)).to(device)

    # 2) 모델 초기화 및 평가 모드 전환
    model = PrototypeMultiModalModel(clinical_dim, image_dim, num_classes).to(device)
    model.eval()  # 필수과정: SHAP/IG 작동을 위해 드롭아웃 등을 동결

    # --------------------------------------------------
    # [파트 A] Integrated Gradients (IG) 연산
    # --------------------------------------------------
    print("\n 2. Captum 라이브러리 기반 Integrated Gradients 연산 시작...")
    ig = IntegratedGradients(model)
    
    # 멀티모달 입력을 튜플로 묶어 전달, 타깃 클래스는 환자들의 가짜 정답 라벨 지정
    ig_attributions, delta = ig.attribute(
        inputs=(dummy_clinical, dummy_image),
        target=dummy_labels,
        return_convergence_delta=True
    )
    
    clinical_ig = ig_attributions[0].cpu().detach().numpy()
    image_ig = ig_attributions[1].cpu().detach().numpy()
    print("IG 추출 성공! (Clinical Shape:", clinical_ig.shape, " / Image Shape:", image_ig.shape, ")")

    # --------------------------------------------------
    # [파트 B] SHAP (DeepExplainer) 연산
    # --------------------------------------------------
    print("\n  3. SHAP 라이브러리 기반 DeepExplainer 연산 시작...")
    # SHAP을 위한 임시 배경(Background) 데이터셋 10개 생성
    bg_clinical = torch.randn(10, clinical_dim).to(device)
    bg_image = torch.randn(10, image_dim).to(device)
    background = [bg_clinical, bg_image]

    explainer = shap.DeepExplainer(model, background)
    
    # 테스트 데이터를 리스트로 묶어 SHAP Value 계산
    test_inputs = [dummy_clinical, dummy_image]
    shap_values = explainer.shap_values(test_inputs)
    
    # 클래스 1(치매 위험군 가정)에 대한 기여도 추출
    clinical_shap = shap_values[1][0]
    image_shap = shap_values[1][1]
    print("SHAP 추출 성공! (Clinical Shape:", clinical_shap.shape, " / Image Shape:", image_shap.shape, ")")

    # --------------------------------------------------
    # [파트 C] 결과 저장 및 시각화 (바탕화면 절대경로 강제 맵핑)
    # --------------------------------------------------
    print("\n 4. 추출된 중요도 데이터 저장 및 시각화 진행...")
    
    # 1. 윈도우 바탕화면 내의 절대 경로 명확하게 지정
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    save_folder = os.path.join(desktop_path, "FlexMoE_Results")

    # 2. 바탕화면에 폴더 생성
    os.makedirs(save_folder, exist_ok=True)

    # 3. 바탕화면 폴더 내부에 넘파이 파일 원본 저장
    np.save(os.path.join(save_folder, "proto_ig_clinical.npy"), clinical_ig)
    np.save(os.path.join(save_folder, "proto_shap_clinical.npy"), clinical_shap)
    print(f" 원본 가중치 데이터(.npy) 저장 완료")

    # 4. 가짜 변수 이름 지정 후 바탕화면 폴더에 차트 이미지 저장
    clinical_names = [f"Clinical_Feature_{i}" for i in range(clinical_dim)]
    
    ig_chart_path = os.path.join(save_folder, "proto_ig_chart.png")
    shap_chart_path = os.path.join(save_folder, "proto_shap_chart.png")
    
    plot_importance(clinical_ig, clinical_names, "Clinical Feature Importance (IG Prototype)", ig_chart_path)
    plot_importance(clinical_shap, clinical_names, "Clinical Feature Importance (SHAP Prototype)", shap_chart_path)

    # 5. 최종 완료 출력
    print(f"\n 바탕화면에 폴더가 생성되었고 파일이 저장되었습니다: {save_folder}")
    print("[성공] 5단계 독립형 프로토타입 코드가 에러 없이 정상 완수되었습니다!")


if __name__ == "__main__":
    main()