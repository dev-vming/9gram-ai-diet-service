import { Feedback } from "./feedback.entity";
import { EntityManager } from "typeorm";
import {
  CheckFeedbackDataDto,
  GetFeedbackDataDto,
  MakeFeedbackDataDto,
} from "./dto/feedback.dto";
import { HttpException } from "@nestjs/common";

export class FeedbackRepository {
  async saveFeedBack(
    feedbackData: MakeFeedbackDataDto,
    manager: EntityManager
  ): Promise<void> {
    try {
      await manager
        .createQueryBuilder(Feedback, "feedback")
        .insert()
        .into(Feedback)
        .values(feedbackData)
        .execute();
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }

  async checkFeedBack(
    checkfeedbackData: CheckFeedbackDataDto,
    manager: EntityManager
  ) {
    try {
      const { userId, question, questionType, feedbackDate } =
        checkfeedbackData;
      const result = await manager
        .createQueryBuilder(Feedback, "feedback")
        .select(["feedback", "feedback_id"])
        .where("feedback.user_id =:userId", { userId })
        .andWhere("feedback.feedback_date =:feedbackDate", {
          feedbackDate,
        })
        .andWhere("feedback.question_type = :questionType", { questionType })
        .andWhere("feedback.question =:question", { question })
        .getOne();
      return result;
    } catch (error) {
      throw error;
    }
  }

  async saveFeedBackYes(feedbackId: string, manager: EntityManager) {
    try {
      return await manager
        .createQueryBuilder(Feedback, "feedback")
        .update(Feedback)
        .set({ saveYn: true })
        .where("feedback.feedback_id = :feedbackId", { feedbackId })
        .execute();
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }

  async getFeedbackData(
    userId: string,
    page: number,
    manager: EntityManager
  ): Promise<GetFeedbackDataDto[]> {
    try {
      return await manager
        .createQueryBuilder(Feedback, "feedback")
        .where("feedback.user_id =:userId", { userId })
        .andWhere("feedback.save_yn =:yn", { yn: true })
        .orderBy("feedback.feedback_date", "DESC")
        .take(5)
        .skip((page - 1) * 5)
        .getMany();
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }

  async getFeedbackChatData(
    userId: string,
    startDate: Date,
    date: Date,
    manager: EntityManager
  ): Promise<GetFeedbackDataDto[]> {
    try {
      return await manager
        .createQueryBuilder(Feedback, "feedback")
        .where("feedback.user_id =:userId", { userId })
        .andWhere("feedback.feedback_date >= :startDate", { startDate })
        .andWhere("feedback.feedback_date <= :date", { date })
        .orderBy("feedback.feedback_date", "ASC")
        .getMany();
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }

  async getFeedbackDetailData(
    feedbackId: string,
    manager: EntityManager
  ): Promise<GetFeedbackDataDto> {
    try {
      return await manager
        .createQueryBuilder(Feedback, "feedback")
        .where("feedback_id =:feedbackId", { feedbackId })
        .getOneOrFail();
    } catch (error) {
      throw error;
    }
  }

  async deleteFeedbackData(feedbackId: string, manager: EntityManager) {
    try {
      return await manager
        .createQueryBuilder(Feedback, "feedback")
        .delete()
        .from(Feedback)
        .where("feedback_id =:feedbackId", { feedbackId })
        .execute();
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }
}
