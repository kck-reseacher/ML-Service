from pydantic import BaseModel, Field
from typing import Optional

class ServingRequestModel(BaseModel):

    class HeaderModel(BaseModel):
        sys_id: str = Field(title="시스템ID", default="102")
        target_id: str = Field(title="타겟ID", default="214")
        inst_type: str = Field(title="인스턴스타입", default="was")
        business_list: Optional[list] = Field(title="비지니스리스트", default="")
        # predict_time: Optional[str] = Field(title="이벤트예측 시간(optional)", default="2022-06-14 10:57:00")
        group_type: Optional[str] = Field(title="그룹 타입", default="instanceGroup")
        group_id: Optional[str] = Field(title="그룹 ID", default="175")

    id1: str = Field(title="모듈명", default="exem_aiops_anls_inst")
    # date: str = Field(title="날짜(YYYYMMDDHHmmss)", default="20220511112700")
    standard_datetime: str = Field(title="날짜(YYYY-MM-DD HH:mm:ss", default="2022-05-11 11:22:00")
    # uid: str = Field(title="uid", default="6f6be211-82ca-4986-b45a-9dde951f518c")
    header: HeaderModel
    body: list = Field(title="서빙 데이터")


class ReloadRequestModel(BaseModel):

    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str

        data: TargetIdModel

    body: DataModel

class InitParamRequestModel(BaseModel):

    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str

        data: TargetIdModel

    body: DataModel

class UpdateDbslnRangeRequestModel(BaseModel):

    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str

        data: TargetIdModel

    body: DataModel

class ReloadOnDeployUserRequestModel(BaseModel):

    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str
            algorithm: str

        data: TargetIdModel

    body: DataModel

class UpdateServiceStatueRequestModel(BaseModel):

    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str
            status: str

        data: TargetIdModel

    body: DataModel

class UpdateSeqattnRangeRequestModel(BaseModel):

    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str

        data: TargetIdModel

    body: DataModel


class UpdateAlgorithmsRangeRequestModel(BaseModel):
    class DataModel(BaseModel):

        class TargetIdModel(BaseModel):
            target_id: str

        data: TargetIdModel

    body: DataModel

class UpdateConfigRequestModel(BaseModel):
    sys_id: str = Field(title="시스템ID", default="102")
    inst_type: str = Field(title="인스턴스타입", default="was")
    target_id: str = Field(title="타겟ID", default="214")
