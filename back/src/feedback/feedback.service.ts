import { Injectable, NotFoundException } from "@nestjs/common";
import { FeedbackRepository } from "./feedback.repository";
import { DataSource } from "typeorm";
import { Feedback } from "./feedback.entity";
import { GetFeedbackDataDto, ResponseDataDto } from "./dto/feedback.dto";
import { plainToInstance } from "class-transformer";
import { CumulativeRecordRepository } from "src/cumulative-record/cumulative.repository";
import { UserRepository } from "src/user/repositories/user.repository";
import { ChatgptApi } from "./utils/chatgpt-api";

@Injectable()
export class FeedbackService {
  constructor(
    private feedBackRepository: FeedbackRepository,
    private readonly cumulativeRepository: CumulativeRecordRepository,
    private readonly userRepository: UserRepository,
    private readonly dataSource: DataSource
  ) {}

  async getFeedbacktoAI(
    userId: string,
    date: Date,
    responseDataDto: ResponseDataDto
  ) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const { question, questionType } = responseDataDto;

      const checkdata = {
        userId,
        question,
        questionType,
        feedbackDate: date,
      };
      const feedbackData = new Feedback().checkfeedbackDataDto(checkdata);
      const beforeData = await this.feedBackRepository.checkFeedBack(
        feedbackData,
        queryRunner.manager
      );
      // 만약 동일 유저의 질문이 있다면, api 호출 x
      if (beforeData) {
        await queryRunner.commitTransaction();
        return {
          feedback: beforeData.feedback,
          feedbackId: beforeData.feedbackId,
        };
      } else {
        // 만약 동일 유저의 질문이 없다면, api 호출
        const totalResult = await this.cumulativeRepository.getDateRecord(
          date,
          userId,
          queryRunner.manager
        );
        const userInfo = await this.userRepository.findUserInfosByUserId(
          userId,
          queryRunner.manager
        );
        // 추후 다른 파일로 빼기

        // ChatGPT API 호출
        const outputText = await ChatgptApi(
          totalResult,
          userInfo,
          questionType
        );

        // const outputText = "어쩌구저쩌구 맛있는 추천";
        const data = {
          userId,
          question,
          questionType,
          feedback: outputText,
          feedbackDate: date,
          saveYn: false,
        };
        const feedbackData = new Feedback().makefeedbackDataDto(data);
        await this.feedBackRepository.saveFeedBack(
          feedbackData,
          queryRunner.manager
        );
        await queryRunner.commitTransaction();
        return { feedback: outputText, feedbackId: feedbackData.feedbackId };
      }
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async saveFeedbackData(feedbackId: string) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const saveReseult = await this.feedBackRepository.saveFeedBackYes(
        feedbackId,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return saveReseult;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async getFeedbackData(
    userId: string,
    page: number
  ): Promise<GetFeedbackDataDto[]> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const result = await this.feedBackRepository.getFeedbackData(
        userId,
        page,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return plainToInstance(GetFeedbackDataDto, result);
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async getFeedbackChatData(
    userId: string,
    startDate: Date,
    date: Date
  ): Promise<GetFeedbackDataDto[]> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const result = await this.feedBackRepository.getFeedbackChatData(
        userId,
        startDate,
        date,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return plainToInstance(GetFeedbackDataDto, result);
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async getFeedbackDetailData(userId: string, feedbackId: string) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const feedbackResult =
        await this.feedBackRepository.getFeedbackDetailData(
          feedbackId,
          queryRunner.manager
        );
      const feedbackResultDto = plainToInstance(
        GetFeedbackDataDto,
        feedbackResult
      );
      if (!feedbackResultDto) {
        throw new NotFoundException("데이터가 존재하지 않습니다");
      }
      if (
        feedbackResultDto.questionType === "목표추천" ||
        (feedbackResultDto.questionType === "식단추천" &&
          feedbackResultDto.question === "내 목표에 맞게 추천받고 싶어!")
      ) {
        const healthInfoResult =
          await this.userRepository.findUserInfosByUserId(
            userId,
            queryRunner.manager
          );
        await queryRunner.commitTransaction();
        return { feedbackResultDto, healthInfoResult };
      } else {
        await queryRunner.commitTransaction();
        return { feedbackResultDto, healthInfoResult: null };
      }
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async deleteFeedbackData(feedbackId: string) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const deleteResult = await this.feedBackRepository.deleteFeedbackData(
        feedbackId,
        queryRunner.manager
      );

      await queryRunner.commitTransaction();
      return deleteResult;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }
}
