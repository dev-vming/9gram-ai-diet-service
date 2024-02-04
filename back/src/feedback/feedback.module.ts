import { Module } from "@nestjs/common";
import { FeedbackController } from "./feedback.controller";
import { FeedbackService } from "./feedback.service";
import { FeedbackRepository } from "./feedback.repository";
import { CumulativeRecordRepository } from "src/cumulative-record/cumulative.repository";
import { UserRepository } from "src/user/repositories/user.repository";

@Module({
  controllers: [FeedbackController],
  providers: [
    FeedbackService,
    FeedbackRepository,
    FeedbackRepository,
    CumulativeRecordRepository,
    UserRepository,
  ],
})
export class FeedbackModule {}
