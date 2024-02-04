import { Injectable, NotFoundException } from "@nestjs/common";
import { Record } from "./record.entity";
import { RecordRepository } from "./record.repository";
import { RecordDto } from "./dtos/record.dto";

@Injectable()
export class RecordService {
  constructor(private recordRepository: RecordRepository) {}

  // 특정 날짜의 식단을 조회하는 메서드
  async getDailyRecord(userId: string, date: string): Promise<Record[]> {
    const records = await this.recordRepository.findByDate(userId, date);
    if (records.length === 0) {
      throw new NotFoundException("해당 날짜의 식단이 없습니다.");
    }
    return records;
  }

  // 식단을 기록하는 메서드
  async createRecord(userId: string, date: string, mealType: number, recordDto: RecordDto): Promise<Record[]> {
    const savedRecord = await this.recordRepository.createOrUpdateRecord(userId, date, mealType, recordDto);
    return savedRecord;
  }

  // 특정 날짜의 식단을 수정하는 메서드
  async updateDailyRecord(userId: string, date: string, mealType: number, recordDto: RecordDto): Promise<Record[]> {
    return await this.recordRepository.createOrUpdateRecord(userId, date, mealType, recordDto);
  }

  // 특정 날짜의 식단을 삭제하는 메서드
  async deleteDailyRecord(userId: string, date: string, mealType: number): Promise<void> {
    await this.recordRepository.deleteRecord(userId,date, mealType);
  }
}