import { FoodInfoAPIRepository } from "./food-info-api.repository";
import { Injectable } from "@nestjs/common";
import { Cron } from "@nestjs/schedule";
import axios from "axios";
import { DataSource } from "typeorm";

@Injectable()
export class FoodInfoApiService {
  private startIdx = 1;
  constructor(
    private foodInfoAPIRepository: FoodInfoAPIRepository,
    private readonly dataSource: DataSource
  ) {}

  // @Cron("30 */2 * * * *")
  // async handleCron() {
  //   try {
  //     await this.getDataPortalProcess();
  //   } catch {
  //     throw new Error();
  //   }
  // }

  async getDataPortal(): Promise<string> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const params = {
        serviceKey: process.env.PORTAL_SERVICE_KEY,
        pageNo: this.startIdx,
        numOfRows: 1000,
        type: "json",
      };
      const url =
        "http://api.data.go.kr/openapi/tn_pubr_public_nutri_food_info_api";
      const res = await axios.get(url, { params: params });
      const data = res.data.response.body.items;
      const result = this.foodInfoAPIRepository.saveDataPortal(
        data,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      if (result) {
        this.startIdx += 1;
        return "성공!";
      } else {
        return "실패";
      }
    } catch (error) {
      await queryRunner.rollbackTransaction();
    } finally {
      await queryRunner.release();
    }
  }

  async getDataPortalProcess(): Promise<string> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const params = {
        serviceKey: process.env.PORTAL_SERVICE_KEY,
        pageNo: this.startIdx,
        numOfRows: 1000,
        type: "json",
      };
      const url =
        "http://api.data.go.kr/openapi/tn_pubr_public_nutri_process_info_api";
      const res = await axios.get(url, { params: params });
      const data = res.data.response.body.items;
      const result = await this.foodInfoAPIRepository.saveDataPortalProcess(
        data,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      if (result) {
        this.startIdx += 1;
        return "성공!";
      } else {
        return "실패";
      }
    } catch (error) {
      await queryRunner.rollbackTransaction();
    } finally {
      await queryRunner.release();
    }
  }
}
