import { ApiProperty } from "@nestjs/swagger";
import { Exclude, Expose, Transform } from "class-transformer";
import { IsDate, IsIn, IsNotEmpty, IsString, IsUUID } from "class-validator";

const QUESTION_TYPES = ["목표추천", "식단추천", "식단평가"];

export class ResponseDataDto {
  @ApiProperty({
    example: "식단평가",
    description: "식단평가 / 식단추천 / 목표추천",
  })
  @IsNotEmpty()
  @IsString()
  @IsIn(QUESTION_TYPES)
  questionType: string;

  @ApiProperty({
    example: "이번주 식단 전체를 평가받을래!",
    description: "AI분석 대화 질문 상세",
  })
  @IsNotEmpty()
  @IsString()
  question: string;
}

@Exclude()
export class MakeFeedbackDataDto {
  @Expose()
  @IsNotEmpty()
  @IsUUID()
  userId: string;

  @Expose()
  @IsNotEmpty()
  @IsString()
  feedback: string;

  @Expose()
  @IsNotEmpty()
  @IsDate()
  feedbackDate: Date;

  @Expose()
  @IsNotEmpty()
  @IsString()
  question: string;

  @Expose()
  @IsNotEmpty()
  @IsString()
  questionType: string;

  @Expose()
  @IsNotEmpty()
  saveYn: boolean;
}

@Exclude()
export class GetFeedbackDataDto {
  @Expose()
  @IsNotEmpty()
  @IsUUID()
  feedbackId: string;

  @Expose()
  @IsNotEmpty()
  @IsString()
  feedback: string;

  @Expose()
  @IsNotEmpty()
  @IsDate()
  feedbackDate: Date;

  @Expose()
  @IsNotEmpty()
  @IsString()
  question: string;

  @Expose()
  @IsNotEmpty()
  @IsString()
  questionType: string;
}

@Exclude()
export class CheckFeedbackDataDto {
  @Expose()
  @IsNotEmpty()
  @IsUUID()
  userId: string;

  @Expose()
  @IsNotEmpty()
  @IsDate()
  feedbackDate: Date;

  @Expose()
  @IsNotEmpty()
  @IsString()
  question: string;

  @Expose()
  @IsNotEmpty()
  @IsString()
  questionType: string;
}
